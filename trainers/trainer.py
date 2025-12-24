# File: trainers/trainer.py (V9.0: Full Integration for Heatmap-Aware Training)

# --- 强制添加项目根目录到 Python 模块搜索路径 ---
import sys
import os

# 获取当前脚本 (trainer.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取项目根目录 (trainer.py 的父目录)
project_root = os.path.dirname(os.path.dirname(current_script_path))
# 将项目根目录插入到 sys.path 的开头
sys.path.insert(0, project_root)

# --- 现在可以安全地导入项目内部模块 ---
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from transformers import BertTokenizer
from data.dataset import PoegraphLayoutDataset, layout_collate_fn 
from models.poem2layout import Poem2LayoutGenerator
from collections import Counter
import numpy as np 
import time
import contextlib 
import torch.nn.utils 
from torch.optim.lr_scheduler import LambdaLR 

# --- NEW IMPORTS for Visualization/Inference/Plotting ---
from inference.greedy_decode import greedy_decode_poem_layout 
from data.visualize import draw_layout
import matplotlib.pyplot as plt 

# --- 导入损失计算函数 ---
from trainers.loss import compute_kl_loss

class LayoutTrainer:
    """负责训练循环、优化器管理、日志记录和模型保存。"""
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        print(f"Trainer initialized on device: {self.device}")
        
        self.tokenizer = tokenizer
        self.example_poem = example_poem
        
        # 训练和保存频率设置
        self.lr = config['training']['learning_rate'] # Store base LR
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.lr
        )
        self.epochs = config['training']['epochs']
        self.output_dir = config['training']['output_dir']
        self.log_steps = config['training'].get('log_steps', 10)
        self.save_every = config['training'].get('save_every', 10)
        self.visualize_every = config['training'].get('visualize_every', 1) 
        
        # 分离绘图路径
        self.plot_path_recons = os.path.join(self.output_dir, "recons_trajectory.png")
        self.plot_path_kl = os.path.join(self.output_dir, "kl_trajectory.png")
        os.makedirs(self.output_dir, exist_ok=True)

        # 学习率调度器初始化
        self.warmup_steps = config['training'].get('warmup_steps', 0)
        self.total_steps = len(train_loader) * self.epochs
        self.global_step = 0
        self.scheduler = self._get_lr_scheduler()

        # 最佳模型路径追踪
        self.current_best_model_path = None
        
        # 损失历史追踪
        self.train_loss_history = []
        self.val_loss_history = []
        
        # 各分量损失历史
        self.val_reg_history = [] 
        self.val_iou_history = []
        self.val_area_history = [] 
        self.val_relation_history = [] 
        self.val_overlap_history = []  
        self.val_size_history = []      
        self.val_kl_history = []      
        
        # 审美损失历史
        self.val_alignment_history = []
        self.val_balance_history = []
        self.val_clustering_history = []
        
        # [NEW V8.0] 视觉态势损失历史
        self.val_gestalt_history = []

    def _get_lr_scheduler(self):
        """定义带线性 Warmup、5 Epoch Hold 和后续衰减的学习率调度器。"""
        N_HOLD_EPOCHS = 5
        steps_per_epoch = len(self.train_loader)
        hold_steps = steps_per_epoch * N_HOLD_EPOCHS
        if self.total_steps < hold_steps: hold_steps = self.total_steps

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            elif current_step < hold_steps:
                return 1.0 
            else:
                decay_start_step = hold_steps
                decay_steps = self.total_steps - decay_start_step
                if decay_steps > 0:
                    relative_step = current_step - decay_start_step
                    return max(0.0, 1.0 - (relative_step / decay_steps))
                return 0.0
        return LambdaLR(self.optimizer, lr_lambda)

    def _update_curriculum(self, epoch):
        """课程学习策略更新。"""
        # --- 策略 A: 权重转移 ---
        if epoch < 50:
            new_rel_weight = 5.0
            new_reg_weight = 1.0
        else:
            transition_start_epoch = 50
            transition_duration = 100 
            progress = min(1.0, (epoch - transition_start_epoch) / transition_duration)
            new_rel_weight = 5.0 - (4.0 * progress) 
            new_reg_weight = 1.0 + (4.0 * progress) 
            
        if hasattr(self.model, 'relation_loss_weight'):
            self.model.relation_loss_weight = new_rel_weight
        if hasattr(self.model, 'reg_loss_weight'):
            self.model.reg_loss_weight = new_reg_weight
        
        # --- 策略 B: KL Annealing ---
        target_kl = 0.005 
        kl_transition_start = 5
        kl_transition_duration = 150 
        
        if epoch < kl_transition_start:
            kl_weight = 0.0
        else:
            kl_progress = min(1.0, (epoch - kl_transition_start) / kl_transition_duration)
            kl_weight = target_kl * kl_progress
            
        return new_rel_weight, new_reg_weight, kl_weight

    def _run_epoch(self, data_loader, is_training: bool, epoch: int = 0):
        self.model.train() if is_training else self.model.eval()
        
        if is_training:
            cur_rel_w, cur_reg_w, cur_kl_w = self._update_curriculum(epoch)
        else:
            cur_rel_w = 5.0; cur_reg_w = 1.0; cur_kl_w = 0.01 

        total_loss_val = 0.0
        t_reg = t_iou = t_area = t_rel = t_over = t_size = 0.0
        t_align = t_bal = t_clus = t_cons = t_gest = 0.0
        t_kl = 0.0
        
        context_manager = contextlib.nullcontext() if is_training else torch.no_grad()
        data_len = len(data_loader)
        
        with context_manager:
            for step, batch in enumerate(data_loader):
                # 1. 移至设备
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                
                # 2. 前向传播 [V9.0 改动：接收 5 个返回值]
                mu, logvar, pred_boxes, decoder_output, pred_heatmaps = self.model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    kg_class_ids=batch['kg_class_ids'], 
                    padding_mask=batch['padding_mask'], 
                    kg_spatial_matrix=batch.get('kg_spatial_matrix'),
                    location_grids=batch.get('location_grids'),
                    target_boxes=batch['target_boxes']
                )
                
                # 3. 计算损失
                loss_tuple = self.model.get_loss(
                    pred_cls=None, pred_bbox_ids=None, pred_boxes=pred_boxes, 
                    pred_count=None, layout_seq=None, layout_mask=batch['loss_mask'], 
                    num_boxes=batch['num_boxes'], target_coords_gt=batch['target_boxes'],
                    kg_spatial_matrix=batch.get('kg_spatial_matrix'),
                    kg_class_weights=batch.get('kg_class_weights'),
                    kg_class_ids=batch['kg_class_ids'], 
                    decoder_output=decoder_output,
                    gestalt_mask=batch.get('gestalt_mask')
                )
                
                # [V9.0 同步解包 12 个损失项]
                (loss_recons, l_rel, l_over, l_reg, l_iou, l_size, l_area, 
                 l_align, l_bal, l_clus, l_cons, l_gestalt) = loss_tuple
                
                # 4. KL 散度
                if mu is not None and logvar is not None:
                    kl_val = compute_kl_loss(mu, logvar, free_bits=1.0)
                else:
                    kl_val = torch.tensor(0.0, device=self.device)
                
                final_loss = loss_recons + cur_kl_w * kl_val
                
                if is_training:
                    self.optimizer.zero_grad()
                    final_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.global_step += 1
                
                # 累加
                total_loss_val += final_loss.item()
                t_reg += l_reg.item(); t_iou += l_iou.item(); t_area += l_area.item()
                t_rel += l_rel.item(); t_over += l_over.item(); t_size += l_size.item()
                t_align += l_align.item(); t_bal += l_bal.item(); t_clus += l_clus.item()
                t_cons += l_cons.item(); t_gest += l_gestalt.item()
                t_kl += kl_val.item()
                
                if is_training and (step + 1) % self.log_steps == 0:
                    print(f"Epoch [{epoch+1}][TRAIN] {step+1}/{data_len} | "
                          f"Tot:{final_loss.item():.3f} | Rel:{l_rel.item():.3f} | "
                          f"Reg:{l_reg.item():.3f} | Gest:{l_gestalt.item():.3f} | KL:{kl_val.item():.3f}")
        
        n = len(data_loader) if len(data_loader) > 0 else 1
        return (total_loss_val/n, t_rel/n, t_over/n, t_reg/n, t_iou/n, t_size/n, t_area/n, 
                t_align/n, t_bal/n, t_clus/n, t_cons/n, t_gest/n, t_kl/n)

    def validate(self, epoch=0):
        start_time = time.time()
        print("\n--- Starting Validation ---")
        
        avg_vals = self._run_epoch(self.val_loader, is_training=False, epoch=epoch)
        (avg_loss, avg_rel, avg_over, avg_reg, avg_iou, avg_size, avg_area, 
         avg_align, avg_bal, avg_clus, avg_cons, avg_gest, avg_kl) = avg_vals
        
        end_time = time.time()
        print(f"--- Validation Finished in {end_time - start_time:.2f}s ---")
        
        # 记录历史
        self.val_loss_history.append(avg_loss)
        self.val_reg_history.append(avg_reg) 
        self.val_iou_history.append(avg_iou)
        self.val_area_history.append(avg_area) 
        self.val_relation_history.append(avg_rel)
        self.val_overlap_history.append(avg_over)
        self.val_size_history.append(avg_size)
        self.val_alignment_history.append(avg_align) 
        self.val_balance_history.append(avg_bal)
        self.val_clustering_history.append(avg_clus)
        self.val_gestalt_history.append(avg_gest)
        self.val_kl_history.append(avg_kl)
        
        print(f"Val Avg: Total:{avg_loss:.4f} | Rel:{avg_rel:.3f} | Over:{avg_over:.3f} | "
              f"Reg:{avg_reg:.3f} | Cons:{avg_cons:.3f} | Gest:{avg_gest:.3f} | KL:{avg_kl:.3f}") 
              
        return avg_loss

    def test(self):
        start_time = time.time()
        print("\n--- Starting Test Set Evaluation ---")
        avg_vals = self._run_epoch(self.test_loader, is_training=False, epoch=999) 
        avg_loss = avg_vals[0]
        print(f"Test Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def _run_inference_example(self, epoch):
        """运行固定样例的推理并保存可视化图片"""
        print(f"\n--- Running Inference Example for Epoch {epoch+1} ---")
        self.model.eval() 
        poem_text = self.example_poem['poem']
        print(f"Poem: {poem_text}")
        
        try:
            ds = self.train_loader.dataset
            if hasattr(ds, 'dataset'): ds = ds.dataset 
            if hasattr(ds, 'pkg'):
                pkg = ds.pkg
                kg_vector = pkg.extract_visual_feature_vector(poem_text)
                indices = torch.nonzero(torch.tensor(kg_vector)).squeeze(1).tolist()
                ids = [i + 2 for i in indices]
                print(f"KG Objects (IDs): {ids}")
        except Exception:
            pass

        max_elements = self.config['model'].get('max_elements', 30)
        
        with torch.no_grad():
            layout = greedy_decode_poem_layout(
                self.model, 
                self.tokenizer, 
                poem_text, 
                max_elements=max_elements,
                device=self.device.type
            )
        
        output_path = os.path.join(self.output_dir, f"epoch_{epoch+1}_layout_pred.png")
        draw_layout(layout, f"PRED (CVAE) E{epoch+1}: {poem_text}", output_path)
        print(f"-> Generated layout saved to {output_path}")

        if epoch == 0 or (epoch + 1) % (self.visualize_every * 10) == 0:
            true_boxes = self.example_poem['boxes']
            true_layout_path = os.path.join(self.output_dir, f"layout_true_example.png")
            draw_layout(true_boxes, f"TRUE: {poem_text}", true_layout_path)
            print(f"-> True layout saved to {true_layout_path}")
            
        print("---------------------------------------------------")
        self.model.train()

    def _plot_loss_history(self):
        """绘制并保存损失变化轨迹图"""
        if not self.train_loss_history: return
        epochs = range(1, len(self.train_loss_history) + 1)
        
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.train_loss_history, label='Train Total', color='blue', marker='o', alpha=0.6)
        plt.plot(epochs, self.val_loss_history, label='Val Total', color='red', marker='s', alpha=0.8)
        
        if len(self.val_reg_history) > 1:
            plt.plot(epochs, self.val_relation_history, label='Val Rel', linestyle=':', alpha=0.7)
            plt.plot(epochs, self.val_overlap_history, label='Val Over', linestyle=':', alpha=0.7)
            plt.plot(epochs, self.val_reg_history, label='Val Reg', linestyle='--', alpha=0.5) 
            plt.plot(epochs, self.val_gestalt_history, label='Val Gestalt', color='black', linewidth=2, linestyle='-')

        plt.title('Loss Trajectory (V9.0: Heatmap & Visual Gestalt)', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        try:
            plt.savefig(self.plot_path_recons) 
            plt.close()
        except Exception as e:
            print(f"[Warning] Could not save Reconstruction loss plot: {e}")

        if len(self.val_kl_history) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, self.val_kl_history, label='Original KL Div', color='darkblue', marker='.', linestyle='-')
            kl_weights = [self._update_curriculum(e - 1)[2] for e in epochs]
            ax2 = plt.gca().twinx()
            ax2.plot(epochs, kl_weights, label='KL Weight', color='red', linestyle='--', alpha=0.5)
            ax2.set_ylabel('KL Weight', color='red')
            plt.title('KL Divergence Trajectory', fontsize=14)
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--')
            try:
                plt.savefig(self.plot_path_kl)
                plt.close()
            except Exception: pass

    def train(self):
        """主训练循环"""
        print("--- Starting Full Training ---")
        best_val_loss = float('inf')
        
        print(f"Total training steps: {self.total_steps}, Warmup steps: {self.warmup_steps}, Base LR: {self.lr:.6e}")
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            print(f"\n==================== Epoch {epoch+1}/{self.epochs} | Training ====================")
            
            avg_train_loss = self._run_epoch(self.train_loader, is_training=True, epoch=epoch)[0]
            self.train_loss_history.append(avg_train_loss)
            
            epoch_end_time = time.time()
            print(f"\nEpoch {epoch+1} finished. Avg Training Loss: {avg_train_loss:.4f} ({epoch_end_time - epoch_start_time:.2f}s)")
            
            avg_val_loss = self.validate(epoch=epoch) 

            self._plot_loss_history()
            
            if (epoch + 1) % self.visualize_every == 0:
                self._run_inference_example(epoch)

            if (epoch + 1) % self.save_every == 0:
                self.test() 
                checkpoint_path = os.path.join(self.output_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(
                    {'model_state_dict': self.model.state_dict(), 
                     'epoch': epoch+1, 
                     'val_loss': avg_val_loss,
                     'optimizer_state_dict': self.optimizer.state_dict()}, 
                    checkpoint_path
                )
                print(f"-> Checkpoint saved to {checkpoint_path}")

            if avg_val_loss < best_val_loss:
                print("-> New best validation loss achieved. Replacing previous best model.")
                if self.current_best_model_path and os.path.exists(self.current_best_model_path):
                    try: os.remove(self.current_best_model_path)
                    except: pass
                
                best_val_loss = avg_val_loss
                model_name = f"model_best_val_loss_{avg_val_loss:.4f}.pth" 
                new_best_path = os.path.join(self.output_dir, model_name)
                
                torch.save(
                    {'model_state_dict': self.model.state_dict(), 
                     'epoch': epoch+1, 
                     'val_loss': avg_val_loss,
                     'optimizer_state_dict': self.optimizer.state_dict()}, 
                    new_best_path
                )
                print(f"-> New best model saved to {new_best_path}")
                self.current_best_model_path = new_best_path
                    
        print("\n--- Training Completed ---")