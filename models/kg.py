# File: tianyusun1/test2/test2-5.1/models/kg.py (V5.11: FIXED LTP UNPACKING BUG)

import torch
import os

# 尝试导入 LTP
try:
    from ltp import LTP
except ImportError:
    print("[Error] LTP package not found. Please run 'pip install ltp'.")
    LTP = None

class PoetryKnowledgeGraph:
    def __init__(self):
        # === 1. 初始化 LTP (加载本地模型) ===
        if LTP is not None:
            # 指定您的本地模型路径
            local_model_path = "/home/610-sty/huggingface/ltp"
            
            print(f"正在加载本地 LTP 模型: {local_model_path} ...")
            try:
                # 直接加载本地路径，不再连接 HuggingFace
                self.ltp = LTP(local_model_path) 
                print("✅ 本地 LTP 模型加载完成。")
            except Exception as e:
                print(f"[Error] 本地 LTP 模型加载失败: {e}")
                print(f"请检查路径 '{local_model_path}' 是否存在且包含模型文件。")
                self.ltp = None
        else:
            self.ltp = None

        # === 2. 基础实体与关系定义 (保持不变) ===
        self.entities = {
            'poem_entity': set(), 'visual_entity': set(), 'scene': set(),
            'spatial_concept': set(), 'attribute': set()
        }
        
        self.relations = {
            'links_to': ('poem_entity', 'visual_entity'),
            'crosses': ('visual_entity', 'visual_entity'),
            'on': ('visual_entity', 'visual_entity'),
            'grows_near': ('visual_entity', 'visual_entity'),
            'in_background_of': ('visual_entity', 'visual_entity'),
            'beside': ('visual_entity', 'visual_entity'),
            'above': ('visual_entity', 'visual_entity'),
            'flows_through': ('visual_entity', 'visual_entity'),
            'surrounds': ('visual_entity', 'visual_entity'),
            'under': ('visual_entity', 'visual_entity'),
            'flies_over': ('visual_entity', 'visual_entity'),
            'grows_under': ('visual_entity', 'visual_entity'),
            'reflects_in': ('visual_entity', 'visual_entity'),
            'descends_to': ('visual_entity', 'visual_entity'),
            'belongs_to': ('poem_entity', 'scene'),
            'implies': ('scene', 'visual_entity'),
            'typically_contains': ('scene', 'visual_entity'),
            'has_attribute': ('visual_entity', 'attribute'),
            'conveys_mood': ('scene', 'attribute')
        }
        
        self.triplets = set()
        
        self.visual_class_mapping = {
            'mountain': 2, 'water': 3, 'people': 4, 'tree': 5,
            'building': 6, 'bridge': 7, 'flower': 8, 'bird': 9, 'animal': 10
        }
        self.num_classes = 9 
        
        self.visual_synonyms = self._init_visual_synonyms()
        self._build_knowledge_graph()
        self._build_lookup_tables()

        # === 3. 数量词典 ===
        self.QUANTITY_MAP = {
            # 数词/量词
            '两': 2, '双': 2, '二': 2, '对': 2, '偶': 2, '并': 2, '一': 1,
            '三': 3, '数': 3, '群': 3, '满': 3, '千': 3, '万': 3, '众': 3, '百': 3, '多': 3, '遍': 3,
            # 形容词/状态词 (针对山水)
            '重': 2, '复': 2 
        }

        # 空间关系与先验
        self.RELATION_IDS = {
            'none': 0, 'above': 1, 'below': 2, 'inside': 3,
            'surrounds': 4, 'on_top': 5, 'near': 6
        }
        
        self.static_priors = {
            (2, 3): 1, (6, 2): 3, (6, 3): 6, (6, 5): 6, (7, 3): 1,
            (5, 2): 3, (5, 3): 6, (8, 5): 6, (8, 2): 3,
            (4, 7): 5, (4, 6): 6, (4, 2): 3, (4, 3): 6, (4, 5): 6,
            (9, 5): 1, (9, 2): 3, (9, 3): 1, (9, 4): 1,
        }

    def _init_visual_synonyms(self):
        return {
            'mountain': ['山', '峰', '峦', '丘', '岭', '岗', '峰峦', '山岭', '山峰', '山丘', '山岗', '崇山', '峻岭', '高山', '远山', '近山', '石', '岩', '径', '小径', '路', '坡', '岸', '沙', '野', '原'],
            'water': ['水', '河', '溪', '江', '湖', '海', '泉', '瀑', '涧', '潭', '池', '波', '浪', '流', '潮', '水波', '流水', '河水', '溪水', '湖水', '江水', '瀑布', '清泉', '深潭', '碧波', '涟漪', '湾', '浦'],
            'people': ['人', '士', '客', '渔', '樵', '僧', '道', '隐', '夫', '女', '童', '子', '行', '旅', '舟子', '渔夫', '樵夫', '隐士', '僧人', '道士', '行人', '旅人', '女子', '童子', '渔人', '老翁', '仙人', '君', '我'],
            'tree': ['树', '木', '松', '柏', '柳', '竹', '梅', '枫', '桂', '梧桐', '杨', '槐', '榆', '樟', '藤', '青松', '翠柏', '垂柳', '翠竹', '梅花', '枫树', '桂树', '梧桐树', '老树', '枯树', '绿树', '花树', '林', '森', '树林', '松阴', '松下'],
            'building': ['屋', '楼', '阁', '亭', '台', '寺', '庙', '庵', '观', '宫', '殿', '宅', '院', '房', '舍', '轩', '榭', '廊', '桥头屋', '茅屋', '草堂', '楼阁', '亭台', '寺庙', '道观', '宫殿', '宅院', '庭院', '廊桥', '水榭', '山亭', '水亭', '画舫', '柴扉', '篱', '墙'],
            'bridge': ['桥', '石桥', '木桥', '拱桥', '曲桥', '栈桥', '浮桥', '廊桥', '竹桥', '独木桥', '石拱桥', '九曲桥', '风雨桥', '虹桥', '小桥', '长桥', '断桥', '古桥', '木拱桥', '石板桥'],
            'flower': ['花', '草', '兰', '菊', '荷', '莲', '蒲', '苇', '菖', '芦', '梅', '桃', '杏', '李', '梨', '樱', '杜鹃', '牡丹', '荷花', '莲花', '兰花', '菊花', '梅花', '桃花', '杏花', '樱花', '杜鹃花', '水仙', '荷花', '芦苇', '蒲草', '芳草', '藤', '蔓', '藤萝', '枝蔓', '苔', '藓', '青苔'],
            'bird': ['鸟', '雀', '燕', '雁', '鹤', '鸥', '鹭', '莺', '鹧', '鸪', '鹊', '鸦', '鹰', '鹏', '凤', '鸾', '孔雀', '白鹭', '黄莺', '杜鹃', '喜鹊', '乌鸦', '大鹏', '仙鹤', '海鸥', '鹧鸪', '沙鸥', '翠鸟', '燕子', '归鸿', '鸿雁', '惊鸿'],
            'animal': ['兽', '鹿', '马', '牛', '虎', '豹', '猿', '猴', '兔', '狐', '狼', '蛇', '鱼', '龙', '凤', '龟', '鹤', '牛', '马', '鹿', '虎', '豹', '猿猴', '白兔', '狐狸', '野狼', '游鱼', '蛟龙', '凤凰', '龟', '仙鹤']
        }

    def _build_knowledge_graph(self):
        self._add_poem_entity_mappings()
        self._add_spatial_relations()
        self._add_scene_themes()
        self._add_attributes()

    def _add_poem_entity_mappings(self):
        mountain_keywords = ['山', '峰', '峦', '丘', '岭', '崖', '岩', '峰峦', '山岭', '崇山', '峻岭', '青山', '远山', '近山', '高山', '云山', '空山', '寒山', '千山', '万壑', '红日', '落日', '斜阳', '夕阳', '朝阳', '残阳', '云', '彩云', '明月', '月', '残月', '新月', '星', '星河', '石', '径', '小径', '幽径', '山径', '坡', '山坡', '岸', '河岸', '沙', '野', '原', '地', '土', '路']
        water_keywords = ['水', '河', '溪', '江', '湖', '海', '泉', '瀑', '涧', '潭', '池', '波', '浪', '流', '潮', '清流', '碧水', '绿水', '秋水', '春水', '流水', '溪流', '江流', '湖水', '河水', '泉水', '瀑布', '湾', '浦']
        people_keywords = ['人', '士', '客', '渔', '樵', '僧', '道', '隐', '夫', '女', '童', '子', '行', '旅', '舟', '渔人', '樵夫', '隐士', '高僧', '道士', '游人', '旅人', '故人', '仙人', '老翁', '童子', '美人', '佳人', '伞', '笠', '琴', '笛', '箫', '酒', '孤舟', '轻舟', '画舫', '客船', '君', '我', '谁']
        tree_keywords = ['树', '木', '松', '柏', '柳', '竹', '梅', '枫', '桂', '梧桐', '杨', '槐', '榆', '樟', '藤', '青松', '翠柏', '垂柳', '翠竹', '梅花', '枫树', '桂树', '梧桐树', '老树', '枯树', '绿树', '花树', '林', '森', '树林', '松阴', '松下']
        flower_keywords = ['花', '草', '兰', '菊', '荷', '莲', '梅', '桃', '杏', '李', '梨', '樱', '杜鹃', '牡丹', '荷花', '莲花', '兰花', '菊花', '梅花', '桃花', '杏花', '樱花', '杜鹃花', '水仙', '芦苇', '蒲草', '芳草', '藤', '蔓', '藤萝', '枝蔓', '苔', '藓', '青苔']
        building_keywords = ['屋', '楼', '阁', '亭', '台', '寺', '庙', '庵', '观', '宫', '殿', '宅', '院', '房', '舍', '轩', '榭', '廊', '桥头屋', '茅屋', '草堂', '楼阁', '亭台', '寺庙', '道观', '宫殿', '宅院', '庭院', '水榭', '柴扉', '扉', '篱', '篱笆', '墙']
        bridge_keywords = ['桥', '石桥', '木桥', '拱桥', '曲桥', '栈桥', '浮桥', '廊桥', '竹桥', '独木桥', '石拱桥', '九曲桥', '风雨桥', '虹桥', '小桥', '长桥', '断桥', '古桥', '木拱桥', '石板桥']
        bird_keywords = ['鸟', '雀', '燕', '雁', '鹤', '鸥', '鹭', '莺', '鹧', '鸪', '鹊', '鸦', '鹰', '鹏', '凤', '鸾', '孔雀', '白鹭', '黄莺', '杜鹃', '喜鹊', '乌鸦', '大鹏', '仙鹤', '海鸥', '鹧鸪', '沙鸥', '翠鸟', '燕子', '归鸿', '鸿雁']
        animal_keywords = ['兽', '鹿', '马', '牛', '虎', '豹', '猿', '猴', '兔', '狐', '狼', '蛇', '鱼', '龙', '凤', '龟', '鹿', '马', '虎', '豹', '猿猴', '白兔', '狐狸', '野狼', '游鱼', '蛟龙', '凤凰', '龟', '仙鹤']
        categories = [
            ('mountain', mountain_keywords), ('water', water_keywords), ('people', people_keywords),
            ('tree', tree_keywords), ('flower', flower_keywords), ('building', building_keywords),
            ('bridge', bridge_keywords), ('bird', bird_keywords), ('animal', animal_keywords)
        ]
        for class_name, keywords in categories:
            for keyword in keywords:
                self._add_triplet(keyword, 'links_to', class_name, 'poem_entity', 'visual_entity')

    def _add_spatial_relations(self):
        self._add_triplet('bridge', 'crosses', 'water', 'visual_entity', 'visual_entity')
        self._add_triplet('stone_bridge', 'crosses', 'river', 'visual_entity', 'visual_entity')
        self._add_triplet('flower', 'surrounds', 'tree', 'visual_entity', 'visual_entity') 
        self._add_triplet('flower', 'on', 'tree', 'visual_entity', 'visual_entity')
        self._add_triplet('mountain', 'in_background_of', 'water', 'visual_entity', 'visual_entity')
        self._add_triplet('peak', 'rises_above', 'river', 'visual_entity', 'visual_entity')
        self._add_triplet('building', 'beside', 'water', 'visual_entity', 'visual_entity')
        self._add_triplet('temple', 'on_mountain', 'mountain', 'visual_entity', 'visual_entity')
        self._add_triplet('cloud', 'above', 'mountain', 'visual_entity', 'visual_entity')
        self._add_triplet('bird', 'flies_over', 'water', 'visual_entity', 'visual_entity')
        self._add_triplet('people', 'walk_on', 'path', 'visual_entity', 'visual_entity')
        self._add_triplet('pagoda', 'stands_on', 'hill', 'visual_entity', 'visual_entity')

    def _add_scene_themes(self):
        scenes = [
            ('misty_mountains', ['空山', '云山', '雾山', '烟山', '山色空蒙', '云雾缭绕']),
            ('riverside_scene', ['小桥流水', '江畔', '水边', '湖畔', '溪畔', '河岸']),
            ('rainy_day', ['雨', '雨亦奇', '细雨', '烟雨', '雨蒙蒙', '雨霏霏']),
            ('spring_scene', ['春', '春色', '春风', '春水', '春山', '春意']),
            ('autumn_scene', ['秋', '秋色', '秋水', '秋山', '秋风', '落叶']),
            ('winter_scene', ['冬', '雪', '冰', '寒山', '雪峰', '雪溪']),
            ('buddhist_retreat', ['寺', '庙', '庵', '僧', '禅', '佛']),
            ('taoist_refuge', ['观', '道', '仙', '隐', '道士', '仙境']),
            ('fishing_scene', ['渔', '舟', '钓', '渔夫', '垂钓', '渔舟']),
            ('woodcutting_scene', ['樵', '伐', '砍', '樵夫', '山林', '伐木']),
            ('scholar_retreat', ['读书', '吟诗', '隐士', '草堂', '书斋', '文人'])
        ]
        for scene_name, keywords in scenes:
            for keyword in keywords:
                self._add_triplet(keyword, 'belongs_to', scene_name, 'poem_entity', 'scene')
        
        scene_implications = {
            'misty_mountains': ['mountain', 'cloud', 'mist', 'path', 'temple'],
            'riverside_scene': ['water', 'bridge', 'tree', 'building', 'people', 'boat', 'flower'],
            'rainy_day': ['cloud', 'rain', 'water', 'umbrella', 'people'],
            'spring_scene': ['tree', 'flower', 'bird', 'butterfly', 'people', 'water'],
            'autumn_scene': ['tree', 'red_leaf', 'mountain', 'water', 'bird', 'people'],
            'winter_scene': ['snow', 'mountain', 'water', 'people', 'building'],
            'buddhist_retreat': ['temple', 'monk', 'mountain', 'path', 'tree', 'bell'],
            'taoist_refuge': ['pavilion', ' Daoist_monk', 'mountain', 'cloud', 'crane', 'immortal'],
            'fishing_scene': ['water', 'boat', 'fisherman', 'rod', 'reeds', 'bird'],
            'woodcutting_scene': ['mountain', 'tree', 'woodcutter', 'axe', 'path', 'forest'],
            'scholar_retreat': ['building', 'scholar', 'book', 'pavilion', 'garden', 'ink', 'brush']
        }
        for scene, entities in scene_implications.items():
            for entity in entities:
                self._add_triplet(scene, 'implies', entity, 'scene', 'visual_entity')
        
        scene_contains = {
            'misty_mountains': ['distant_mountains', 'clouds', 'winding_path', 'hidden_temple'],
            'riverside_scene': ['curved_bridge', 'rippling_water', 'willow_trees', 'pavilion_by_water', 'fishing_boat'],
            'rainy_day': ['rain_streaks', 'distant_mountains', 'misty_water', 'people_with_umbrellas', 'dripping_eaves'],
            'spring_scene': ['cherry_blossoms', 'swallows', 'frogs', 'new_green_leaves', 'butterflies', 'gentle_breeze'],
            'autumn_scene': ['maple_trees_with_red_leaves', 'geese_flying_south', 'harvest_fields', 'moon_reflection_on_water'],
            'winter_scene': ['snow_covered_mountains', 'frozen_lake', 'bare_trees', 'people_in_thick_clothes', 'warm_cottage'],
            'buddhist_retreat': ['ancient_temple', 'stone_stairs', 'bell_tower', 'praying_monks', 'incense_smoke'],
            'taoist_refuge': ['mountain_pavilion', 'crane', 'pine_trees', 'flowing_robes', 'mysterious_clouds', 'immortal_figures'],
            'fishing_scene': ['bamboo_pole', 'reeds_along_shore', 'ripples_on_water', 'distant_mountains', 'heron_watching'],
            'woodcutting_scene': ['axe', 'cut_logs', 'dense_forest', 'mountain_path', 'carrying_firewood', 'waterfall_nearby'],
            'scholar_retreat': ['scrolls', 'ink_stone', 'brush', 'lute', 'crane_standing_by_window', 'moon_viewing_pavilion']
        }
        for scene, elements in scene_contains.items():
            for element in elements:
                self._add_triplet(scene, 'typically_contains', element, 'scene', 'visual_entity')

    def _add_attributes(self):
        visual_attributes = {
            'mountain': ['tall', 'majestic', 'distant', 'misty', 'steep', 'green', 'snowy', 'ancient'],
            'water': ['clear', 'rippled', 'flowing', 'still', 'deep', 'shallow', 'broad', 'narrow'],
            'people': ['small', 'distant', 'walking', 'sitting', 'fishing', 'rowing', 'meditating', 'contemplating'],
            'tree': ['tall', 'ancient', 'gnarled', 'green', 'red_autumn_leaves', 'bare_winter', 'blooming', 'dense_foliage'],
            'building': ['small', 'ancient', 'wooden', 'stone', 'painted_roof', 'open_window', 'hanging_lantern', 'enclosed_courtyard'],
            'bridge': ['curved', 'arched', 'stone', 'wood', 'simple', 'ornate', 'weathered', 'covered'],
            'flower': ['blossoming', 'dewy', 'colorful', 'delicate', 'wild_growing', 'planted_in_garden', 'fragrant', 'scattered_petals'],
            'bird': ['flying', 'perched', 'calling', 'flock', 'solitary', 'white_crane', 'golden_oriole', 'wild_goose_in_flight'],
            'animal': ['grazing', 'drinking', 'hiding', 'running', 'peaceful', 'majestic_deer', 'white_rabbit', 'monkeys_in_trees']
        }
        for entity, attrs in visual_attributes.items():
            for attr in attrs:
                self._add_triplet(entity, 'has_attribute', attr, 'visual_entity', 'attribute')
        
        scene_moods = {
            'misty_mountains': ['tranquil', 'mysterious', 'majestic', 'contemplative', 'ethereal'],
            'riverside_scene': ['peaceful', 'harmonious', 'leisurely', 'refreshing', 'melancholy'],
            'rainy_day': ['melancholic', 'tranquil', 'refreshing', 'cool', 'introspective'],
            'spring_scene': ['joyful', 'hopeful', 'vibrant', 'renewal', 'gentle'],
            'autumn_scene': ['nostalgic', 'mellow', 'reflective', 'serene', 'bittersweet'],
            'winter_scene': ['serene', 'pure', 'isolated', 'contemplative', 'resilient'],
            'buddhist_retreat': ['tranquil', 'serene', 'detached', 'spiritual', 'meditative'],
            'taoist_refuge': ['mysterious', 'eternal', 'natural', 'free', 'transcendent'],
            'fishing_scene': ['leisurely', 'peaceful', 'self_sufficient', 'harmonious', 'simple'],
            'woodcutting_scene': ['industrious', 'rugged', 'natural', 'practical', 'connected_to_earth'],
            'scholar_retreat': ['refined', 'thoughtful', 'cultured', 'contemplative', 'aesthetic']
        }
        for scene, moods in scene_moods.items():
            for mood in moods:
                self._add_triplet(scene, 'conveys_mood', mood, 'scene', 'attribute')

    def _add_triplet(self, head, relation, tail, head_type=None, tail_type=None):
        self.triplets.add((head, relation, tail))
        if head_type: self.entities[head_type].add(head)
        if tail_type: self.entities[tail_type].add(tail)

    def _build_lookup_tables(self):
        self.keyword_to_class_id = {}
        self.scene_to_implied_ids = {}
        for head, rel, tail in self.triplets:
            if rel == 'links_to':
                cls_id = self.get_visual_class_id(tail)
                if cls_id:
                    self.keyword_to_class_id[head] = cls_id
            elif rel == 'belongs_to':
                if head not in self.scene_to_implied_ids:
                    self.scene_to_implied_ids[head] = set()
                implied_elements = self.get_scene_elements(tail)
                for elem in implied_elements:
                    cls_id = self.get_visual_class_id(elem)
                    if cls_id:
                        self.scene_to_implied_ids[head].add(cls_id)

    def get_visual_class_id(self, visual_entity):
        if visual_entity in self.visual_class_mapping:
            return self.visual_class_mapping[visual_entity]
        return None

    def get_scene_elements(self, scene_name):
        elements = set()
        for head, rel, tail in self.triplets:
            if head == scene_name and (rel == 'implies' or rel == 'typically_contains'):
                elements.add(tail)
        return elements

    def extract_visual_feature_vector(self, poem_text: str) -> torch.Tensor:
        visual_vector = torch.zeros(self.num_classes)
        for keyword, cls_id in self.keyword_to_class_id.items():
            if keyword in poem_text:
                idx = cls_id - 2
                if 0 <= idx < self.num_classes:
                    visual_vector[idx] = 1.0
        
        for keyword, implied_ids in self.scene_to_implied_ids.items():
            if keyword in poem_text:
                for cls_id in implied_ids:
                    idx = cls_id - 2
                    if 0 <= idx < self.num_classes:
                        if visual_vector[idx] < 1.0:
                            visual_vector[idx] = 0.5 
        return visual_vector

    def extract_spatial_matrix(self, poem_text: str) -> torch.Tensor:
        num_classes = 9
        relation_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
        feature_vec = self.extract_visual_feature_vector(poem_text)
        present_indices = [i for i, val in enumerate(feature_vec) if val > 0]
        
        for idx_a in present_indices:
            for idx_b in present_indices:
                if idx_a == idx_b: continue
                cls_a = idx_a + 2
                cls_b = idx_b + 2
                if (cls_a, cls_b) in self.static_priors:
                    relation_matrix[idx_a, idx_b] = self.static_priors[(cls_a, cls_b)]

        if '高' in poem_text and '树' in poem_text:
            tree_idx = 5 - 2 
            if tree_idx in present_indices:
                for other_idx in present_indices:
                    if other_idx != tree_idx and (other_idx + 2) != 2: 
                        relation_matrix[tree_idx, other_idx] = self.RELATION_IDS['above']

        if ('幽' in poem_text or '深' in poem_text) and '山' in poem_text:
            mtn_idx = 2 - 2 
            if mtn_idx in present_indices:
                for other_idx in present_indices:
                    if other_idx != mtn_idx:
                        relation_matrix[other_idx, mtn_idx] = self.RELATION_IDS['inside']

        boat_words = ['舟', '船', '舫']
        if any(w in poem_text for w in boat_words):
            ppl_idx = 4 - 2 
            water_idx = 3 - 2 
            if ppl_idx in present_indices and water_idx in present_indices:
                relation_matrix[ppl_idx, water_idx] = self.RELATION_IDS['inside']

        sky_words = ['月', '星', '日', '阳']
        if any(w in poem_text for w in sky_words):
            mtn_idx = 2 - 2 
            if mtn_idx in present_indices:
                for other_idx in present_indices:
                    if other_idx != mtn_idx:
                        relation_matrix[mtn_idx, other_idx] = self.RELATION_IDS['above']

        if '绕' in poem_text or '缠' in poem_text or '围' in poem_text:
            flower_idx = 8 - 2 
            tree_idx = 5 - 2 
            if flower_idx in present_indices and tree_idx in present_indices:
                relation_matrix[flower_idx, tree_idx] = self.RELATION_IDS['surrounds']
            water_idx = 3 - 2 
            bldg_idx = 6 - 2 
            mtn_idx = 2 - 2 
            if water_idx in present_indices:
                if bldg_idx in present_indices:
                    relation_matrix[water_idx, bldg_idx] = self.RELATION_IDS['surrounds']
                if mtn_idx in present_indices:
                    relation_matrix[water_idx, mtn_idx] = self.RELATION_IDS['surrounds']

        if '满' in poem_text or '遍' in poem_text:
            flower_idx = 8 - 2
            mtn_idx = 2 - 2 
            if flower_idx in present_indices and mtn_idx in present_indices:
                relation_matrix[flower_idx, mtn_idx] = self.RELATION_IDS['inside']

        if '松下' in poem_text or '树下' in poem_text or '林下' in poem_text:
            tree_idx = 5 - 2
            if tree_idx in present_indices:
                for other_idx in present_indices:
                    if other_idx != tree_idx:
                        if (other_idx + 2) != 9: 
                            relation_matrix[other_idx, tree_idx] = self.RELATION_IDS['below']

        fly_over_words = ['飞过', '掠', '渡', '过', '翔']
        if any(w in poem_text for w in fly_over_words):
            bird_idx = 9 - 2 
            if bird_idx in present_indices:
                for other_idx in present_indices:
                    if (other_idx + 2) in [3, 2]:
                        relation_matrix[bird_idx, other_idx] = self.RELATION_IDS['above']

        return relation_matrix

    # [MODIFIED V5.11] 增强鲁棒性的 LTP 依存解析 (修复解包错误)
    def expand_ids_with_quantity(self, unique_class_ids, poem):
        """
        使用 LTP 分析句子依存关系，精准判断数量修饰对象。
        增加了对 deps 格式的兼容性检查。
        """
        if self.ltp is None:
            return unique_class_ids

        # 1. LTP 推理
        try:
            output = self.ltp.pipeline([poem], tasks=["cws", "pos", "dep"])
            words = output.cws[0] 
            deps = output.dep[0]  
        except Exception:
            return unique_class_ids

        # 2. 找到物体词索引
        object_locs = {}
        for cid in unique_class_ids:
            object_locs[cid] = []
            
        for i, w in enumerate(words):
            for kw, cid in self.keyword_to_class_id.items():
                if cid in unique_class_ids and kw in w: 
                    object_locs[cid].append(i)
                    break 

        # 3. 依存树搜索
        final_counts = {cid: 1 for cid in unique_class_ids}
        debug_info = []

        for cid, word_indices in object_locs.items():
            best_multiplier = 1
            
            for obj_idx in word_indices:
                # 遍历所有词的依存关系
                for mod_idx, dep_item in enumerate(deps):
                    
                    # [FIX] 兼容性处理：不同版本的 LTP 返回格式可能不同 (tuple vs dict)
                    if isinstance(dep_item, dict):
                        head_idx = dep_item.get('head', 0)
                        label = dep_item.get('label', '')
                    elif isinstance(dep_item, (list, tuple)) and len(dep_item) >= 2:
                        head_idx = dep_item[0]
                        label = dep_item[1]
                    else:
                        continue # 跳过无法解析的格式

                    # head_idx 在 LTP 中是 1-based
                    if head_idx == 0: continue
                    parent = head_idx - 1
                    
                    # Case 1: Modifier -> Head (例如: "两" -> "鸟", label='ATT'/'QUN')
                    if parent == obj_idx:
                        modifier_word = words[mod_idx]
                        if modifier_word in self.QUANTITY_MAP:
                            best_multiplier = max(best_multiplier, self.QUANTITY_MAP[modifier_word])
                        
                        # 递归检查子节点
                        for sub_mod_idx, sub_item in enumerate(deps):
                            # [FIX] 同样的兼容性处理
                            if isinstance(sub_item, dict):
                                sub_head = sub_item.get('head', 0)
                            elif isinstance(sub_item, (list, tuple)) and len(sub_item) >= 1:
                                sub_head = sub_item[0]
                            else:
                                continue
                                
                            if sub_head - 1 == mod_idx:
                                sub_word = words[sub_mod_idx]
                                if sub_word in self.QUANTITY_MAP:
                                    best_multiplier = max(best_multiplier, self.QUANTITY_MAP[sub_word])

                    # Case 2: Subject -> Head (例如: "山" -> "重", label='SBV')
                    if mod_idx == obj_idx: 
                        head_word = words[parent]
                        if head_word in self.QUANTITY_MAP:
                            if label == 'SBV':
                                best_multiplier = max(best_multiplier, self.QUANTITY_MAP[head_word])

            final_counts[cid] = max(final_counts[cid], best_multiplier)

        # 4. 生成结果
        expanded_ids = []
        for cid, count in final_counts.items():
            count = min(count, 3) 
            if count > 1:
                debug_info.append(f"Class {cid} x{count}")
            for _ in range(count):
                expanded_ids.append(cid)
                
        if debug_info:
            # 简化打印，避免干扰进度条
            # print(f"  [KG-LTP] '{poem[:10]}...' -> {', '.join(debug_info)}")
            pass
            
        return expanded_ids

    def visualize(self):
        pass 

if __name__ == "__main__":
    pkg = PoetryKnowledgeGraph()
    test_poems = [
        "两只黄鹂鸣翠柳", 
        "山重水复疑无路",
        "柳暗花明又一村"
    ]
    
    print("\n--- Testing LTP Logic ---")
    if pkg.ltp:
        for p in test_poems:
            vec = pkg.extract_visual_feature_vector(p)
            ids = [i+2 for i, v in enumerate(vec) if v > 0]
            expanded = pkg.expand_ids_with_quantity(ids, p)
            print(f"Poem: {p} -> IDs: {ids} -> Expanded: {expanded}")
    else:
        print("LTP not loaded, skipping tests.")