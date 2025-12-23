import sys
import os
import torch
import random

# === 关键：添加项目根目录到 sys.path，以便导入 models.kg ===
# 假设当前文件在 stage2_generation/utils/ 下，根目录在 ../../
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from models.kg import PoetryKnowledgeGraph
except ImportError:
    print("[Warning] Could not import models.kg. Make sure you are in the project root.")
    PoetryKnowledgeGraph = None

class KGPromptGenerator:
    """
    创新点组件：基于 KG 的意境 Prompt 生成器
    
    功能：
    不仅提取诗句中的"物体"(Entities)，还通过 KG 推理出隐含的"意境"(Moods)和"场景"(Scenes)。
    构建结构化的 Prompt 传给 SDXL。
    """
    def __init__(self):
        print("[KGPrompt] Initializing Knowledge Graph...")
        if PoetryKnowledgeGraph is not None:
            self.pkg = PoetryKnowledgeGraph()
            # 建立反向查找表，加速推理
            self._build_inference_maps()
        else:
            self.pkg = None
            
        # 基础风格 Prompt (你可以根据需要调整)
        self.base_style = (
            "Chinese traditional ink wash painting, masterpiece, highly detailed, "
            "elegant, atmospheric, texturing with brush strokes, negative space, "
            "watercolor style, artstation"
        )

    def _build_inference_maps(self):
        """
        从 pkg.triplets 中构建 (Keyword -> Scene) 和 (Scene -> Mood) 的映射
        """
        self.keyword_to_scene = {}
        self.scene_to_moods = {}
        
        # 遍历所有三元组，提取知识
        for head, rel, tail in self.pkg.triplets:
            # 关系1: 关键词 -> 场景 (e.g., "空山" -> "misty_mountains")
            if rel == 'belongs_to':
                if head not in self.keyword_to_scene:
                    self.keyword_to_scene[head] = set()
                self.keyword_to_scene[head].add(tail)
            
            # 关系2: 场景 -> 意境 (e.g., "misty_mountains" -> "tranquil")
            elif rel == 'conveys_mood':
                if head not in self.scene_to_moods:
                    self.scene_to_moods[head] = set()
                self.scene_to_moods[head].add(tail)

    def generate_prompt(self, poem_text):
        """
        输入: 诗句 (str)
        输出: 增强后的 Prompt (str)
        """
        if self.pkg is None:
            return f"{self.base_style}, {poem_text}"

        # 1. 提取视觉实体 (Entities)
        # 复用 pkg 的向量提取功能
        visual_vec = self.pkg.extract_visual_feature_vector(poem_text)
        present_indices = torch.nonzero(visual_vec > 0).squeeze(1).tolist()
        
        entities = []
        # 将 Class ID 转回英文单词 (2=mountain, 3=water...)
        # 这里我们需要一个简单的映射，或者直接硬编码
        id_to_name = {
            0: "mountain", 1: "water", 2: "people", 3: "tree",
            4: "building", 5: "bridge", 6: "flower", 7: "bird", 8: "animal"
        }
        for idx in present_indices:
            if idx in id_to_name:
                entities.append(id_to_name[idx])
        
        # 2. 推理意境与场景 (Moods & Scenes)
        found_scenes = set()
        found_moods = set()
        
        # 简单的关键词匹配
        for keyword, scenes in self.keyword_to_scene.items():
            if keyword in poem_text:
                found_scenes.update(scenes)
                
        for scene in found_scenes:
            if scene in self.scene_to_moods:
                found_moods.update(self.scene_to_moods[scene])

        # 3. 组装 Prompt
        # 结构: [Style] + [Subject/Entities] + [Atmosphere/Moods] + [Poem Content]
        
        prompt_parts = [self.base_style]
        
        if entities:
            entity_str = "featuring " + ", ".join(entities)
            prompt_parts.append(entity_str)
            
        if found_moods:
            # 随机取 3 个意境词，避免 Prompt 太长
            mood_list = list(found_moods)
            random.shuffle(mood_list)
            mood_str = "atmosphere: " + ", ".join(mood_list[:3])
            prompt_parts.append(mood_str)
        
        # 加入原诗作为一种"概念指引"
        # 注意：SDXL 对中文理解一般，所以主要靠前面的英文关键词，加入中文是为了保留一点语义
        prompt_parts.append(f"concept: {poem_text}")

        full_prompt = ", ".join(prompt_parts)
        return full_prompt

# 简单的测试代码
if __name__ == "__main__":
    generator = KGPromptGenerator()
    
    test_poems = [
        "空山新雨后",        # 应该触发 'misty_mountains', 'rainy_day' -> 'tranquil', 'refreshing'
        "孤舟蓑笠翁",        # 应该触发 'fishing_scene' -> 'solitary', 'peaceful'
        "日照香炉生紫烟"      # 应该触发 'mountain' 
    ]
    
    print("\n--- Testing Prompt Generation ---")
    for poem in test_poems:
        prompt = generator.generate_prompt(poem)
        print(f"\nPoem: {poem}")
        print(f"Prompt: {prompt}")