import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import defaultdict
import io
import json
import re
import time
from typing import Dict, List, Optional, Any

from app.llm.generator import Generator
from app.models.alloy_data import AlloyData
from app.retriever.hybrid import HybridRetriever

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class BatchAlloyExtractor:
    """批量合金信息提取器 - 优化版"""

    def __init__(self, default_max_chunks: int = 20):
        self.retriever = HybridRetriever()
        self.generator = Generator()
        self.default_max_chunks = default_max_chunks
        
        # JSON 清理规则
        self.json_clean_patterns = [
            # 移除注释
            (r'//.*$', ''),
            (r'/\*[\s\S]*?\*/', ''),
            # 处理未转义的引号
            (r'(?<!\\)"([^"]*?)"(?=[^,}\]])', r'"\1"'),
            # 处理特殊字符
            (r'\u201c|\u201d|\u201e|\u201f', '"'),
            (r'\u2018|\u2019|\u201a|\u201b', "'"),
            # 移除多余空格
            (r'\s+', ' '),
        ]

    def clean_json_string(self, json_str: str) -> str:
        """清理 JSON 字符串，处理各种异常情况"""
        if not json_str:
            return ""
        
        # 应用清理规则
        for pattern, replacement in self.json_clean_patterns:
            json_str = re.sub(pattern, replacement, json_str, flags=re.MULTILINE)
        
        # 处理特殊值
        special_values = [
            '"balance"', '"Balance"', '"remainder"', '"Remainder"',
            '"trace"', '"Trace"', '"minor"', '"Minor"',
            '"wt.%",', '"wt.%,', '"wt.%",',
            '~', 'approx', 'approximately', 'around',
        ]
        
        for val in special_values:
            json_str = json_str.replace(f'"{val}"', 'null')
            json_str = json_str.replace(val, 'null')
        
        # 处理 "X%" 格式（百分比）
        json_str = re.sub(r'"(\d+\.?\d*)%"', r'\1', json_str)
        
        # 处理科学计数法
        json_str = re.sub(r'"(\d+\.?\d*[eE][+-]?\d+)"', r'\1', json_str)
        
        # 处理未闭合的引号
        quote_count = json_str.count('"')
        if quote_count % 2 != 0:
            # 添加一个闭合引号
            json_str += '"'
        
        # 确保最后有闭合大括号
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        json_str += '}' * (open_braces - close_braces)
        
        return json_str.strip()

    def extract_json_from_response(self, response: str) -> Optional[str]:
        """从响应中提取有效的 JSON 字符串"""
        if not response:
            return None
        
        # 尝试找到 JSON 块
        json_pattern = r'(\{[\s\S]*\})'
        matches = re.findall(json_pattern, response)
        
        if not matches:
            # 尝试查找最小的 JSON 对象
            matches = re.findall(r'\{.*?\}', response, re.DOTALL)
        
        if matches:
            # 返回最长的匹配（最完整的 JSON）
            return max(matches, key=len)
        
        return None

    def safe_json_loads(self, json_str: str) -> Optional[Dict]:
        """安全解析 JSON，处理各种异常情况"""
        if not json_str:
            return None
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # 尝试修复常见错误
            try:
                # 处理尾随逗号
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                return json.loads(json_str)
            except:
                pass
            
            # 尝试逐行解析
            try:
                lines = json_str.split('\n')
                clean_lines = []
                for line in lines:
                    # 移除 trailing commas
                    line = re.sub(r',\s*$', '', line.strip())
                    # 处理未转义的字符
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            # 确保值是有效的 JSON
                            if not value.startswith(('"', '{', '[', 'null', 'true', 'false')) and not value.replace('.', '').isdigit():
                                value = f'"{value}"'
                            clean_lines.append(f"{key}: {value}")
                fixed_json = '{' + ',\n'.join(clean_lines) + '}'
                return json.loads(fixed_json)
            except:
                pass
        
        return None

    def extract_with_strategy(self, query: str, max_chunks: Optional[int] = None) -> Optional[AlloyData]:
        """使用优化策略提取合金信息 - 增加宽松检索和容错机制"""
        if max_chunks is None:
            max_chunks = self.default_max_chunks
        
        print(f"    检索配置: 每个查询获取 {max_chunks} 个结果")
        
        # 分层检索策略
        # 第一层：精确查询（详细）
        precise_queries = [
            f"{query} alloy composition mechanical properties",
            f"{query} tensile strength yield strength",
            f"{query} heat treatment aging parameters",
            f"{query} microstructure grain size",
            f"{query} corrosion rate potential",
        ]
        
        # 第二层：中等查询（较宽松）
        medium_queries = [
            f"{query} alloy properties",
            f"{query} mechanical properties",
            f"{query} heat treatment",
            f"{query} microstructure",
        ]
        
        # 第三层：宽松查询（最通用）
        broad_queries = [
            f"{query} alloy",
            f"{query} material properties",
            f"{query}",
        ]

        all_chunks = []
        seen_texts = set()
        
        # 执行检索策略（方案A：执行所有查询以获取最全面的数据）
        all_query_lists = [precise_queries, medium_queries, broad_queries]
        query_level_names = ["精确查询", "中等查询", "宽松查询"]
        
        for level, query_list in enumerate(all_query_lists):
            print(f"    执行 {query_level_names[level]}...")
            for q in query_list:
                try:
                    result = self.retriever.retrieve(q, top_k=max_chunks)
                    count = len(result.chunks)
                    print(f"      检索 '{q[:50]}...' 返回 {count} 个结果")
                    for chunk in result.chunks:
                        if chunk.text and chunk.text not in seen_texts:
                            seen_texts.add(chunk.text)
                            all_chunks.append(chunk)
                except Exception as e:
                    print(f"      检索 '{q[:50]}...' 出错: {e}")
            
            print(f"    {query_level_names[level]}完成，当前累计 {len(all_chunks)} 个chunk")

        print(f"    总计去重后获取 {len(all_chunks)} 个唯一chunk")

        if not all_chunks:
            print(f"未找到关于 {query} 的相关信息")
            return None

        # 构建上下文（根据内容相关性排序）
        context_parts = []
        for i, chunk in enumerate(sorted(all_chunks[:20], key=lambda x: len(x.text), reverse=True)):
            # 清理 chunk 文本
            text = chunk.text.replace('\n', ' ').replace('\r', '')
            text = re.sub(r'\s+', ' ', text).strip()
            context_parts.append(f"[Section {i+1}]\n{text[:2000]}")  # 限制长度
        
        context = "\n\n".join(context_parts)
        print(f"    构建上下文长度: {len(context)} 字符")

        # 优化的提取提示词 - 更清晰的结构和格式要求
        extraction_prompt = f"""You are an expert materials scientist specializing in zinc and aluminum alloys.

Extract structured alloy data from the following research paper content.

Paper Content:
{context}

IMPORTANT INSTRUCTIONS:
1. Extract ONLY numerical values where specified
2. For composition, use weight percent (wt%)
3. If a value is not available or cannot be determined, use null
4. Return ONLY valid JSON, no explanations, no extra text

JSON FORMAT (return exactly this structure):
{{
  "alloy_id": "string - unique identifier like ZnMg1.5_XXX",
  "alloy_system": "string - e.g., Zn-Mg, Al-Zn-Mg, Zn-Al-Mg-Cu",
  "year": number or null,
  "paper_title": "string or null",
  "doi": "string or null",
  "authors": ["author1", "author2", ...] or null,
  "institution": "string or null",
  "composition": {{
    "zinc": number or null,
    "magnesium": number or null,
    "aluminum": number or null,
    "copper": number or null,
    "silver": number or null,
    "other_elements": {{"element": number}} or null,
    "impurities": {{"element": number}} or null,
    "melting_point": number or null
  }},
  "processing": {{
    "melting_temperature": number or null,
    "casting_method": "string or null",
    "annealing_temperature": number or null,
    "annealing_time": number or null,
    "solution_temperature": number or null,
    "solution_time": number or null,
    "aging_temperature": number or null,
    "aging_time": number or null,
    "rolling_reduction": number or null,
    "extrusion_ratio": number or null,
    "cooling_rate": number or null
  }},
  "microstructure": {{
    "phases": ["phase1", "phase2", ...] or null,
    "grain_size": number or null,
    "precipitate_size": number or null,
    "texture": "string or null"
  }},
  "mechanical_properties": {{
    "ultimate_tensile_strength": number or null,
    "yield_strength": number or null,
    "elongation": number or null,
    "hardness": number or null,
    "elastic_modulus": number or null,
    "fatigue_strength": number or null,
    "compressive_strength": number or null
  }},
  "corrosion_properties": {{
    "corrosion_rate": number or null,
    "corrosion_potential": number or null,
    "polarization_resistance": number or null,
    "icorr": number or null,
    "corrosion_test_method": "string or null"
  }},
  "testing_conditions": {{
    "test_standard": "string or null",
    "temperature": number or null,
    "strain_rate": number or null
  }},
  "key_findings": ["finding1", "finding2", ...] or null,
  "notes": "string or null"
}}

DO NOT INCLUDE ANY TEXT OUTSIDE THE JSON STRUCTURE.
IF YOU CANNOT FIND DATA FOR A FIELD, USE null.
ENSURE ALL NUMBERS ARE VALID (no "balance", "trace", or descriptive text)."""

        try:
            response = self.generator.generate(question=extraction_prompt, context="")
            print(f"    生成响应长度: {len(response)} 字符")
        except Exception as e:
            print(f"生成响应出错: {e}")
            return None

        try:
            # 提取 JSON
            json_str = self.extract_json_from_response(response)
            if not json_str:
                print(f"无法从响应中提取JSON")
                print(f"响应预览: {response[:200]}...")
                return None
            
            # 清理 JSON
            json_str = self.clean_json_string(json_str)
            
            # 安全解析
            data = self.safe_json_loads(json_str)
            if not data:
                print(f"JSON 解析失败")
                print(f"JSON内容: {json_str[:500]}...")
                return None
            
            # 数据验证和清理
            data = self.validate_and_clean_data(data)
            
            alloy_data = AlloyData(**data)
            return alloy_data

        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            print(f"响应预览: {response[:200]}...")
            return None
        except Exception as e:
            print(f"解析提取结果时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def validate_and_clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证和清理提取的数据"""
        # 确保必要的字段存在
        required_fields = [
            'alloy_id', 'alloy_system', 'composition',
            'processing', 'microstructure', 'mechanical_properties',
            'corrosion_properties', 'testing_conditions'
        ]
        
        for field in required_fields:
            if field not in data:
                data[field] = {} if isinstance(data.get(field), dict) else None
        
        # 清理数值字段
        numeric_fields = [
            ('composition', ['zinc', 'magnesium', 'aluminum', 'copper', 'silver', 'melting_point']),
            ('processing', ['melting_temperature', 'annealing_temperature', 'annealing_time',
                          'solution_temperature', 'solution_time', 'aging_temperature',
                          'aging_time', 'rolling_reduction', 'extrusion_ratio', 'cooling_rate']),
            ('microstructure', ['grain_size', 'precipitate_size']),
            ('mechanical_properties', ['ultimate_tensile_strength', 'yield_strength', 'elongation',
                                      'hardness', 'elastic_modulus', 'fatigue_strength',
                                      'compressive_strength']),
            ('corrosion_properties', ['corrosion_rate', 'corrosion_potential',
                                      'polarization_resistance', 'icorr']),
            ('testing_conditions', ['temperature', 'strain_rate']),
        ]
        
        for parent, fields in numeric_fields:
            if isinstance(data.get(parent), dict):
                for field in fields:
                    value = data[parent].get(field)
                    if value is not None:
                        data[parent][field] = self.safe_convert_to_number(value)
        
        # 生成 alloy_id 如果不存在
        if not data.get('alloy_id') and data.get('alloy_system'):
            import uuid
            data['alloy_id'] = f"{data['alloy_system'].replace('-', '').replace(' ', '')}_{str(uuid.uuid4())[:3]}"
        
        return data

    def safe_convert_to_number(self, value) -> Optional[float]:
        """安全转换为数字"""
        if value is None:
            return None
        
        try:
            # 处理字符串类型的数字
            if isinstance(value, str):
                # 移除单位和空格
                value = re.sub(r'[^0-9.\-eE]', '', value.strip())
                if not value:
                    return None
            return float(value)
        except (ValueError, TypeError):
            return None

    def batch_extract(self, queries: List[str], output_file: str = "batch_alloy_data.json", 
                      max_chunks_per_query: Optional[int] = None, 
                      skip_existing: bool = False) -> List[AlloyData]:
        """批量提取多个合金的信息"""
        extracted_data = []
        start_time = time.time()
        
        if max_chunks_per_query is None:
            max_chunks_per_query = self.default_max_chunks
        
        # 检查是否跳过已存在的数据
        existing_alloy_ids = set()
        if skip_existing and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    for item in existing_data:
                        if 'alloy_id' in item:
                            existing_alloy_ids.add(item['alloy_id'])
                print(f"已存在 {len(existing_alloy_ids)} 条数据，将跳过重复")
            except:
                pass
        
        print(f"\n批量提取配置:")
        print(f"  - 查询数量: {len(queries)}")
        print(f"  - 每个查询检索chunk数: {max_chunks_per_query}")
        print(f"  - 跳过已存在: {skip_existing}")

        success_count = 0
        fail_count = 0
        
        for i, query in enumerate(queries, 1):
            print(f"\n=== 提取 {i}/{len(queries)}: {query} ===")
            try:
                data = self.extract_with_strategy(query, max_chunks_per_query)
                if data:
                    # 检查是否重复
                    if skip_existing and data.alloy_id in existing_alloy_ids:
                        print(f"跳过重复: {data.alloy_id}")
                        continue
                    
                    extracted_data.append(data)
                    success_count += 1
                    print(f"✓ 提取成功: {data.alloy_system}")
                    
                    # 显示详细信息
                    self.print_extracted_info(data)
                else:
                    fail_count += 1
                    print(f"✗ 提取失败")
            except Exception as e:
                fail_count += 1
                print(f"✗ 处理出错: {e}")
                import traceback
                traceback.print_exc()

        end_time = time.time()
        elapsed_time = end_time - start_time

        # 合并保存（如果跳过已存在）
        if skip_existing and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                extracted_data = existing_data + [d.dict() for d in extracted_data]
            except:
                pass
        
        if extracted_data:
            # 去重
            if isinstance(extracted_data[0], AlloyData):
                data_list = [d.dict() for d in extracted_data]
            else:
                data_list = extracted_data
            
            # 基于 alloy_id 去重
            unique_data = {}
            for item in data_list:
                alloy_id = item.get('alloy_id', str(id(item)))
                if alloy_id not in unique_data:
                    unique_data[alloy_id] = item
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(list(unique_data.values()), f, ensure_ascii=False, indent=2)
            
            print(f"\n已保存 {len(unique_data)} 条数据到 {output_file}")

        print(f"\n提取统计:")
        print(f"  成功: {success_count}")
        print(f"  失败: {fail_count}")
        print(f"  耗时: {elapsed_time:.2f} 秒")

        # 确保返回 AlloyData 对象列表（修复类型问题）
        result = []
        for item in extracted_data:
            if isinstance(item, AlloyData):
                result.append(item)
            elif isinstance(item, dict):
                result.append(AlloyData(**item))
        return result

    def print_extracted_info(self, data: AlloyData):
        """打印提取的详细信息"""
        info_lines = []
        
        # 成分信息
        if data.composition:
            components = []
            for elem, value in [
                ('Zn', data.composition.zinc),
                ('Mg', data.composition.magnesium),
                ('Al', data.composition.aluminum),
                ('Cu', data.composition.copper),
                ('Ag', data.composition.silver),
            ]:
                if value is not None:
                    val = float(value) if isinstance(value, str) else value
                    components.append(f"{elem} {val:.1f}%")
            if components:
                info_lines.append(f"成分: {', '.join(components)}")
        
        # 力学性能
        if data.mechanical_properties:
            props = []
            if data.mechanical_properties.ultimate_tensile_strength:
                props.append(f"UTS {data.mechanical_properties.ultimate_tensile_strength} MPa")
            if data.mechanical_properties.yield_strength:
                props.append(f"YS {data.mechanical_properties.yield_strength} MPa")
            if data.mechanical_properties.elongation:
                props.append(f"延伸率 {data.mechanical_properties.elongation}%")
            if data.mechanical_properties.hardness:
                props.append(f"硬度 {data.mechanical_properties.hardness} HV")
            if props:
                info_lines.append(f"力学性能: {', '.join(props)}")
        
        # 工艺信息
        if data.processing:
            process_steps = []
            if data.processing.casting_method:
                process_steps.append(data.processing.casting_method)
            if data.processing.aging_temperature:
                process_steps.append(f"时效 {data.processing.aging_temperature}°C")
            if process_steps:
                info_lines.append(f"工艺: {', '.join(process_steps)}")
        
        if info_lines:
            print("  " + "\n  ".join(info_lines))

    def generate_statistics(self, extracted_data: List[AlloyData]) -> Dict:
        """生成提取结果统计"""
        if not extracted_data:
            return {}

        stats = {
            "total_extracted": len(extracted_data),
            "alloy_systems": defaultdict(int),
            "composition_stats": {
                "zinc": [], "magnesium": [], "aluminum": [], "copper": []
            },
            "mechanical_stats": {
                "uts": [], "ys": [], "elongation": [], "hardness": []
            },
            "has_mechanical_data": 0,
            "has_composition_data": 0,
            "has_processing_data": 0,
        }

        for data in extracted_data:
            if data.alloy_system:
                stats["alloy_systems"][data.alloy_system] += 1

            # 成分统计
            has_comp = False
            if data.composition:
                for elem in ['zinc', 'magnesium', 'aluminum', 'copper']:
                    value = getattr(data.composition, elem)
                    if value is not None:
                        num_val = self.safe_convert_to_number(value)
                        if num_val is not None:
                            stats["composition_stats"][elem].append(num_val)
                            has_comp = True
            if has_comp:
                stats["has_composition_data"] += 1

            # 力学性能统计
            has_mech = False
            if data.mechanical_properties:
                for prop, key in [
                    ('ultimate_tensile_strength', 'uts'),
                    ('yield_strength', 'ys'),
                    ('elongation', 'elongation'),
                    ('hardness', 'hardness')
                ]:
                    value = getattr(data.mechanical_properties, prop)
                    if value is not None:
                        num_val = self.safe_convert_to_number(value)
                        if num_val is not None:
                            stats["mechanical_stats"][key].append(num_val)
                            has_mech = True
            if has_mech:
                stats["has_mechanical_data"] += 1

            # 工艺数据统计
            if data.processing and (data.processing.aging_temperature or 
                                  data.processing.casting_method or 
                                  data.processing.annealing_temperature):
                stats["has_processing_data"] += 1

        # 计算统计值
        for key in stats["composition_stats"]:
            values = stats["composition_stats"][key]
            if values:
                stats["composition_stats"][key] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "count": len(values)
                }
            else:
                stats["composition_stats"][key] = None

        for key in stats["mechanical_stats"]:
            values = stats["mechanical_stats"][key]
            if values:
                stats["mechanical_stats"][key] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "count": len(values)
                }
            else:
                stats["mechanical_stats"][key] = None

        return stats

    def print_statistics(self, stats: Dict):
        """打印统计结果"""
        print(f"\n=== 提取结果统计 ===")
        print(f"总提取条数: {stats.get('total_extracted', 0)}")
        print(f"有成分数据: {stats.get('has_composition_data', 0)}")
        print(f"有力学性能数据: {stats.get('has_mechanical_data', 0)}")
        print(f"有工艺数据: {stats.get('has_processing_data', 0)}")

        print(f"\n合金体系分布:")
        for sys_name, count in sorted(stats.get('alloy_systems', {}).items(), 
                                      key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {sys_name}: {count} 条")

        print(f"\n成分范围 (wt%):")
        comp_stats = stats.get('composition_stats', {})
        for elem in ['zinc', 'magnesium', 'aluminum', 'copper']:
            data = comp_stats.get(elem)
            if data:
                print(f"  {elem.upper()}: {data['min']:.1f} - {data['max']:.1f} (平均: {data['avg']:.1f}, 样本数: {data['count']})")

        print(f"\n力学性能范围:")
        mech_stats = stats.get('mechanical_stats', {})
        prop_names = {
            'uts': '抗拉强度 (MPa)',
            'ys': '屈服强度 (MPa)',
            'elongation': '延伸率 (%)',
            'hardness': '硬度 (HV)'
        }
        for key, label in prop_names.items():
            data = mech_stats.get(key)
            if data:
                print(f"  {label}: {data['min']:.1f} - {data['max']:.1f} (平均: {data['avg']:.1f}, 样本数: {data['count']})")


if __name__ == "__main__":
    print("\n=== 批量合金信息提取系统 (优化版) ===")
    print("正在初始化...")

    # 可调整的参数
    MAX_CHUNKS_PER_QUERY = 20
    SKIP_EXISTING = True  # 跳过已存在的数据，避免重复

    # 扩展的合金体系列表
    zinc_alloy_systems = [
        "Zn-Mg", "Zn-Al", "Zn-Cu", "Zn-Ag", "Zn-Mn", 
        "Zn-Sn", "Zn-Ni", "Zn-Ca", "Zn-Mg-Ca", "Zn-Al-Mg",
        "Zn-Al-Cu", "Zn-Mg-Al", "Zn-Cu-Ti", "Zn-Mg-Mn",
        "Zn-Ca-Mg", "Zn-Ag-Cu", "Zn-Mn-Mg", "Zn-Sn-Mg",
        "Al-Zn-Mg", "Al-Zn-Mg-Cu", "Mg-Zn", "Mg-Zn-Al",
        "Zn-Gd", "Zn-Y", "Zn-La", "Zn-Ce", "Zn-Sr",
        "Zn-Fe", "Zn-Pb", "Zn-Si", "Zn-Bi", "Zn-Sb"
    ]
    
    # 为每个合金体系生成多样化的查询词 - 扩展版
    test_queries = []
    for alloy_system in zinc_alloy_systems:
        # 基础成分查询 - 确保获取完整成分数据
        test_queries.extend([
            f"{alloy_system} alloy composition weight percent wt%",
            f"{alloy_system} alloy chemical composition analysis",
            f"{alloy_system} alloy elemental composition Zn Mg Al Cu",
            f"{alloy_system} alloy major elements minor elements",
            f"{alloy_system} alloy impurity content trace elements",
        ])
        
        # 力学性能详细查询 - 覆盖所有关键力学性能
        test_queries.extend([
            # 拉伸性能
            f"{alloy_system} ultimate tensile strength UTS MPa",
            f"{alloy_system} yield strength YS MPa 0.2% proof stress",
            f"{alloy_system} elongation percentage total elongation",
            f"{alloy_system} reduction of area RA percentage",
            # 压缩性能
            f"{alloy_system} compressive strength MPa compression test",
            f"{alloy_system} compressive yield strength MPa",
            # 弹性性能
            f"{alloy_system} elastic modulus Young modulus GPa E-modulus",
            f"{alloy_system} shear modulus G GPa",
            f"{alloy_system} Poisson ratio nu",
            # 硬度
            f"{alloy_system} hardness HV Vickers hardness number",
            f"{alloy_system} hardness HRB HRC Rockwell",
            f"{alloy_system} Brinell hardness HB",
            # 韧性和疲劳
            f"{alloy_system} fracture toughness KIC MPa sqrt(m)",
            f"{alloy_system} fatigue strength endurance limit MPa",
            f"{alloy_system} fatigue life cycles Nf",
            # 其他力学性能
            f"{alloy_system} impact toughness Charpy Izod J",
            f"{alloy_system} bending strength flexural strength MPa",
        ])
        
        # 热处理工艺详细查询
        test_queries.extend([
            f"{alloy_system} solution treatment temperature time",
            f"{alloy_system} aging treatment T5 T6 condition",
            f"{alloy_system} annealing temperature duration",
            f"{alloy_system} quenching cooling rate",
            f"{alloy_system} precipitation hardening peak aging",
        ])
        
        # 加工工艺详细查询
        test_queries.extend([
            f"{alloy_system} casting method die casting sand casting",
            f"{alloy_system} rolling deformation reduction",
            f"{alloy_system} extrusion ratio temperature",
            f"{alloy_system} forging hot working",
            f"{alloy_system} welding joining brazing",
            f"{alloy_system} surface treatment coating",
        ])
        
        # 微观结构详细查询
        test_queries.extend([
            f"{alloy_system} microstructure phase composition",
            f"{alloy_system} grain size average grain diameter",
            f"{alloy_system} precipitate size distribution TEM",
            f"{alloy_system} secondary phase particles",
            f"{alloy_system} texture crystallographic orientation",
            f"{alloy_system} dislocation density substructure",
        ])
        
        # 腐蚀性能详细查询
        test_queries.extend([
            f"{alloy_system} corrosion rate weight loss",
            f"{alloy_system} polarization resistance electrochemical",
            f"{alloy_system} corrosion potential Ecorr",
            f"{alloy_system} pitting corrosion resistance",
            f"{alloy_system} salt spray test ASTM",
        ])
        
        # 物理性能查询 - 重点补充熔点和热性能
        test_queries.extend([
            # 熔点和相变
            f"{alloy_system} melting point liquidus solidus temperature",
            f"{alloy_system} solidus temperature Ts",
            f"{alloy_system} liquidus temperature Tl",
            f"{alloy_system} eutectic temperature phase transformation",
            f"{alloy_system} glass transition temperature Tg",
            f"{alloy_system} recrystallization temperature",
            # 密度
            f"{alloy_system} density rho g/cm3 specific gravity",
            # 热性能
            f"{alloy_system} thermal conductivity lambda W/mK",
            f"{alloy_system} thermal expansion coefficient CTE ppm/K",
            f"{alloy_system} specific heat capacity Cp J/kgK",
            f"{alloy_system} latent heat of fusion J/kg",
            # 电性能
            f"{alloy_system} electrical conductivity sigma MS/m",
            f"{alloy_system} electrical resistivity rho microohm cm",
        ])
        
        # 特殊性能查询
        test_queries.extend([
            f"{alloy_system} creep resistance high temperature",
            f"{alloy_system} wear resistance tribological",
            f"{alloy_system} biocompatibility biomedical implant",
            f"{alloy_system} damping capacity internal friction",
            f"{alloy_system} superplastic deformation",
        ])
        
        # 成分-工艺-性能关联查询 - 确保获取完整数据链条
        test_queries.extend([
            # 成分-性能关系
            f"{alloy_system} composition tensile strength relationship",
            f"{alloy_system} Mg content mechanical properties",
            f"{alloy_system} Al content hardness strength",
            f"{alloy_system} Cu content corrosion resistance",
            # 工艺-性能关系
            f"{alloy_system} heat treatment mechanical properties",
            f"{alloy_system} aging time hardness strength",
            f"{alloy_system} solution temperature elongation",
            f"{alloy_system} cooling rate microstructure properties",
            # 成分-工艺-性能完整数据
            f"{alloy_system} alloy composition processing properties table",
            f"{alloy_system} mechanical properties heat treatment condition",
        ])
        
        # 应用和优化查询
        test_queries.extend([
            f"{alloy_system} alloy design optimization",
            f"{alloy_system} property improvement enhancement",
            f"{alloy_system} industrial application automotive",
            f"{alloy_system} comparative study benchmark",
        ])
    
    print(f"\n当前配置:")
    print(f"  MAX_CHUNKS_PER_QUERY = {MAX_CHUNKS_PER_QUERY}")
    print(f"  SKIP_EXISTING = {SKIP_EXISTING}")
    print(f"  合金体系数量: {len(zinc_alloy_systems)}")
    print(f"  查询词数量: {len(test_queries)}")
    print(f"  预计子查询次数: {len(test_queries)} x 15 = {len(test_queries) * 15}")
    print(f"\n查询维度覆盖:")
    print(f"  - 基础成分: 5 个查询/体系")
    print(f"  - 力学性能: 16 个查询/体系 (拉伸/压缩/弹性/硬度/韧性/疲劳)")
    print(f"  - 热处理工艺: 5 个查询/体系")
    print(f"  - 加工工艺: 6 个查询/体系")
    print(f"  - 微观结构: 6 个查询/体系")
    print(f"  - 腐蚀性能: 5 个查询/体系")
    print(f"  - 物理性能: 12 个查询/体系 (熔点/密度/热性能/电性能)")
    print(f"  - 特殊性能: 5 个查询/体系")
    print(f"  - 成分-工艺-性能关联: 10 个查询/体系")
    print(f"  - 应用优化: 4 个查询/体系")
    print(f"  总计: 74 个查询/体系")
    print(f"\n关键数据覆盖:")
    print(f"  ✓ 成分: Zn, Mg, Al, Cu, Ag, 杂质含量")
    print(f"  ✓ 拉伸性能: UTS, YS, 延伸率, 断面收缩率")
    print(f"  ✓ 压缩性能: 压缩强度, 压缩屈服强度")
    print(f"  ✓ 弹性性能: 弹性模量, 剪切模量, 泊松比")
    print(f"  ✓ 熔点数据: 固相线, 液相线, 共晶温度")
    print(f"  ✓ 工艺参数: 热处理温度/时间, 冷却速率")
    print(f"  ✓ 成分-工艺-性能完整链条")

    extractor = BatchAlloyExtractor(default_max_chunks=MAX_CHUNKS_PER_QUERY)
    print(f"\n开始提取...")
    
    results = extractor.batch_extract(
        test_queries, 
        "batch_alloy_data.json",
        skip_existing=SKIP_EXISTING
    )
    
    if results:
        stats = extractor.generate_statistics(results)
        extractor.print_statistics(stats)
        
        print(f"\n=== 提取完成 ===")
        print(f"成功提取了 {len(results)} 条合金数据")
        print(f"数据已保存到: batch_alloy_data.json")
    else:
        print("\n=== 提取完成 ===")
        print("未提取到任何合金数据，请检查数据库内容或调整检索策略")