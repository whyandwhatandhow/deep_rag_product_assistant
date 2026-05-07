import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.retriever.hybrid import HybridRetriever
from app.llm.generator import Generator
from app.models.alloy_data import AlloyData
from typing import Optional, List, Dict
import json
import time
from collections import defaultdict

class BatchAlloyExtractor:
    """批量合金信息提取器"""

    def __init__(self, default_max_chunks: int = 25):
        self.retriever = HybridRetriever()
        self.generator = Generator()
        self.default_max_chunks = default_max_chunks

    def extract_with_strategy(self, query: str, max_chunks: Optional[int] = None) -> Optional[AlloyData]:
        """使用优化策略提取合金信息
        
        Args:
            query: 检索查询词
            max_chunks: 每个查询检索的chunk数量，默认使用实例的default_max_chunks
        """
        if max_chunks is None:
            max_chunks = self.default_max_chunks
        
        print(f"    检索配置: 每个查询获取 {max_chunks} 个结果")
        
        retrieval_queries = [
            f"{query} alloy composition mechanical properties",
            f"{query} tensile strength yield strength microstructure",
            f"{query} heat treatment aging process parameters",
            f"{query} corrosion resistance electrochemical properties",
            f"{query} phase composition grain size precipitates",
            f"{query} creep fatigue wear properties",  # 新增
            f"{query} processing route manufacturing method",  # 新增
            f"{query} biocompatibility biodegradation",  # 新增
            f"{query} thermal conductivity electrical properties",  # 新增
            f"{query} welding joining brazing"  # 新增
        ]

        all_chunks = []
        seen_texts = set()

        for q in retrieval_queries:
            try:
                result = self.retriever.retrieve(q, top_k=max_chunks)
                print(f"    检索 '{q[:50]}...' 返回 {len(result.chunks)} 个结果")
                for chunk in result.chunks:
                    if chunk.text not in seen_texts:
                        seen_texts.add(chunk.text)
                        all_chunks.append(chunk)
            except Exception as e:
                print(f"    检索出错: {e}")
                continue

        print(f"    总计去重后获取 {len(all_chunks)} 个唯一chunk")

        if not all_chunks:
            print(f"未找到关于 {query} 的相关信息")
            return None

        # 构建上下文
        context_parts = []
        for i, chunk in enumerate(all_chunks[:15]):  # 增加上下文长度
            context_parts.append(f"[Section {i+1}]{chunk.text}")
        context = "\n\n".join(context_parts)

        # 提取提示词 - 英文内容用英文提取更准确
        extraction_prompt = f"""You are a materials science expert specializing in zinc and aluminum alloys.

From the following research paper content, extract structured data about the alloy.

Paper Content:
{context}

Extract the following in JSON format ONLY (return valid JSON, no explanations):
{{
  "alloy_id": "unique identifier based on composition (e.g., ZnMg1.5_001)",
  "alloy_system": "alloy system type (e.g., Zn-Mg, Al-Zn-Mg, Zn-Al-Mg)",
  "year": publication year if mentioned,
  "paper_title": "paper title if visible",
  "doi": "DOI if mentioned",
  "authors": ["author1", "author2"] if mentioned,
  "institution": "institution if mentioned",
  "composition": {{
    "zinc": Zn content in wt%,
    "magnesium": Mg content in wt% (if applicable),
    "aluminum": Al content in wt% (if applicable),
    "copper": Cu content in wt% (if applicable),
    "silver": Ag content in wt% (if applicable),
    "other_elements": {{"element": "content"}} (if applicable),
    "impurities": {{"element": "content"}} (if applicable),
    "composition_range": {{"element": {{"min": value, "max": value}}}} (if mentioned),
    "melting_point": melting point of the alloy in °C (if mentioned)
  }},
  "processing": {{
    "melting_temperature": temperature in °C,
    "melting_time": time in min,
    "melting_atmosphere": "atmosphere type",
    "casting_method": "method used",
    "cooling_rate": cooling rate in °C/min,
    "mold_temperature": temperature in °C,
    "rolling_reduction": reduction in %, 
    "extrusion_ratio": extrusion ratio,
    "deformation_temperature": temperature in °C,
    "deformation_rate": rate in s⁻¹,
    "annealing_temperature": temperature in °C,
    "annealing_time": time in hours,
    "solution_temperature": temperature in °C,
    "solution_time": time in hours,
    "aging_temperature": temperature in °C,
    "aging_time": time in hours,
    "surface_treatment": "treatment method",
    "process_route": ["step1", "step2"]
  }},
  "microstructure": {{
    "phases": ["phase1", "phase2"],
    "phase_fraction": {{"phase1": fraction}},
    "grain_size": size in μm if mentioned,
    "grain_size_distribution": {{"range": fraction}},
    "precipitates": "precipitate description",
    "precipitate_size": size in nm,
    "precipitate_density": density in per μm³,
    "texture": "texture description",
    "dislocation_density": density in m⁻²
  }},
  "mechanical_properties": {{
    "ultimate_tensile_strength": UTS in MPa,
    "yield_strength": yield strength in MPa,
    "elongation": elongation in %,
    "hardness": hardness in HV,
    "elastic_modulus": modulus in GPa,
    "poisson_ratio": ratio,
    "fatigue_strength": strength in MPa,
    "impact_energy": energy in J,
    "compressive_strength": strength in MPa,
    "shear_strength": strength in MPa,
    "properties_at_temperature": {{temperature: {{"property": value}}}}
  }},
  "corrosion_properties": {{
    "corrosion_rate": rate in mm/year,
    "corrosion_potential": potential in mV,
    "polarization_resistance": resistance in Ω·cm²,
    "icorr": current density in μA/cm²,
    "corrosion_test_method": "test method",
    "test_medium": "test medium"
  }},
  "testing_conditions": {{
    "test_standard": "standard name",
    "temperature": temperature in °C,
    "humidity": humidity in %,
    "strain_rate": rate in s⁻¹,
    "test_equipment": "equipment name",
    "loading_type": "loading type",
    "test_duration": duration
  }},
  "key_findings": ["main finding 1", "main finding 2"],
  "optimization_goals": ["goal 1", "goal 2"],
  "simulation_parameters": {{"parameter": "value"}},
  "experimental_verification": {{"result": "value"}},
  "notes": "additional notes"
}}

If a field cannot be determined from the content, use null. Return ONLY JSON."""

        try:
            response = self.generator.generate(question=extraction_prompt, context="")
        except Exception as e:
            print(f"生成响应出错: {e}")
            return None

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                print(f"无法从响应中提取JSON")
                return None

            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            alloy_data = AlloyData(**data)
            return alloy_data

        except Exception as e:
            print(f"解析提取结果时出错: {e}")
            print(f"响应内容: {response[:500]}...")
            return None

    def batch_extract(self, queries: List[str], output_file: str = "batch_alloy_data.json", 
                      max_chunks_per_query: Optional[int] = None) -> List[AlloyData]:
        """批量提取多个合金的信息
        
        Args:
            queries: 需要提取的合金查询列表
            output_file: 输出文件路径
            max_chunks_per_query: 每个查询检索的最大chunk数，默认使用实例的default_max_chunks
        """
        extracted_data = []
        start_time = time.time()
        
        if max_chunks_per_query is None:
            max_chunks_per_query = self.default_max_chunks
        
        print(f"\n批量提取配置:")
        print(f"  - 查询数量: {len(queries)}")
        print(f"  - 每个查询检索chunk数: {max_chunks_per_query}")
        print(f"  - 总查询次数: {len(queries)} x 10个子查询 = {len(queries) * 10}")
        print(f"  - 预计最大chunk数: {len(queries) * 10 * max_chunks_per_query}")

        for i, query in enumerate(queries, 1):
            print(f"\n=== 提取 {i}/{len(queries)}: {query} ===")
            try:
                data = self.extract_with_strategy(query, max_chunks_per_query)
                if data:
                    extracted_data.append(data)
                    print(f"提取成功: {data.alloy_system}")
                    
                    # 显示详细信息
                    if data.composition:
                        components = []
                        if data.composition.zinc:
                            components.append(f"Zn-{data.composition.zinc:.1f}%")
                        if data.composition.magnesium:
                            components.append(f"Mg-{data.composition.magnesium:.1f}%")
                        if data.composition.aluminum:
                            components.append(f"Al-{data.composition.aluminum:.1f}%")
                        if data.composition.copper:
                            components.append(f"Cu-{data.composition.copper:.1f}%")
                        if data.composition.other_elements:
                            for elem, val in data.composition.other_elements.items():
                                components.append(f"{elem}-{val}")
                        if components:
                            print(f"成分: {', '.join(components)}")
                    
                    if data.mechanical_properties:
                        props = []
                        if data.mechanical_properties.ultimate_tensile_strength:
                            props.append(f"UTS: {data.mechanical_properties.ultimate_tensile_strength} MPa")
                        if data.mechanical_properties.yield_strength:
                            props.append(f"YS: {data.mechanical_properties.yield_strength} MPa")
                        if data.mechanical_properties.elongation:
                            props.append(f"延伸率: {data.mechanical_properties.elongation}%")
                        if data.mechanical_properties.hardness:
                            props.append(f"硬度: {data.mechanical_properties.hardness} HV")
                        if props:
                            print(f"力学性能: {', '.join(props)}")
                    
                    if data.processing:
                        process_steps = []
                        if data.processing.casting_method:
                            process_steps.append(data.processing.casting_method)
                        if data.processing.annealing_temperature:
                            process_steps.append(f"退火 {data.processing.annealing_temperature}°C")
                        if data.processing.aging_temperature:
                            process_steps.append(f"时效 {data.processing.aging_temperature}°C")
                        if process_steps:
                            print(f"工艺: {', '.join(process_steps)}")
                else:
                    print(f"提取失败")
            except Exception as e:
                print(f"处理出错: {e}")

        end_time = time.time()
        elapsed_time = end_time - start_time

        if extracted_data:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([d.dict() for d in extracted_data], f, ensure_ascii=False, indent=2)
            print(f"\n已保存 {len(extracted_data)} 条数据到 {output_file}")

        return extracted_data

    def generate_statistics(self, extracted_data: List[AlloyData]) -> Dict:
        """生成提取结果统计"""
        if not extracted_data:
            return {}

        stats = {
            "total_extracted": len(extracted_data),
            "alloy_systems": defaultdict(int),
            "composition_stats": {
                "zinc_range": [],
                "magnesium_range": [],
                "aluminum_range": [],
                "copper_range": []
            },
            "mechanical_stats": {
                "uts_range": [],
                "ys_range": [],
                "elongation_range": [],
                "hardness_range": []
            },
            "processing_methods": defaultdict(int),
            "microstructure_phases": defaultdict(int)
        }

        for data in extracted_data:
            # 合金体系统计
            if data.alloy_system:
                stats["alloy_systems"][data.alloy_system] += 1

            # 成分统计
            if data.composition:
                if data.composition.zinc:
                    stats["composition_stats"]["zinc_range"].append(data.composition.zinc)
                if data.composition.magnesium:
                    stats["composition_stats"]["magnesium_range"].append(data.composition.magnesium)
                if data.composition.aluminum:
                    stats["composition_stats"]["aluminum_range"].append(data.composition.aluminum)
                if data.composition.copper:
                    stats["composition_stats"]["copper_range"].append(data.composition.copper)

            # 力学性能统计
            if data.mechanical_properties:
                if data.mechanical_properties.ultimate_tensile_strength:
                    stats["mechanical_stats"]["uts_range"].append(data.mechanical_properties.ultimate_tensile_strength)
                if data.mechanical_properties.yield_strength:
                    stats["mechanical_stats"]["ys_range"].append(data.mechanical_properties.yield_strength)
                if data.mechanical_properties.elongation:
                    stats["mechanical_stats"]["elongation_range"].append(data.mechanical_properties.elongation)
                if data.mechanical_properties.hardness:
                    stats["mechanical_stats"]["hardness_range"].append(data.mechanical_properties.hardness)

            # 工艺方法统计
            if data.processing:
                if data.processing.casting_method:
                    stats["processing_methods"][data.processing.casting_method] += 1

            # 微观结构相统计
            if data.microstructure and data.microstructure.phases:
                for phase in data.microstructure.phases:
                    stats["microstructure_phases"][phase] += 1

        # 计算范围和平均值 - 先收集数据，再批量更新避免迭代中修改字典
        comp_stats_updates = {}
        for key, values in stats["composition_stats"].items():
            if values and isinstance(values, list):
                comp_stats_updates[f"{key}_min"] = min(values)
                comp_stats_updates[f"{key}_max"] = max(values)
                comp_stats_updates[f"{key}_avg"] = sum(values) / len(values)
        stats["composition_stats"].update(comp_stats_updates)

        mech_stats_updates = {}
        for key, values in stats["mechanical_stats"].items():
            if values and isinstance(values, list):
                mech_stats_updates[f"{key}_min"] = min(values)
                mech_stats_updates[f"{key}_max"] = max(values)
                mech_stats_updates[f"{key}_avg"] = sum(values) / len(values)
        stats["mechanical_stats"].update(mech_stats_updates)

        return stats

    def print_statistics(self, stats: Dict):
        """打印统计结果"""
        print(f"\n=== 提取结果统计 ===")
        print(f"总提取条数: {stats.get('total_extracted', 0)}")

        print(f"\n合金体系分布:")
        for sys, count in stats.get('alloy_systems', {}).items():
            print(f"  {sys}: {count} 条")

        print(f"\n成分范围:")
        comp_stats = stats.get('composition_stats', {})
        if comp_stats.get('zinc_range'):
            print(f"  Zn: {comp_stats.get('zinc_range_min', 0):.1f}% - {comp_stats.get('zinc_range_max', 0):.1f}% (平均: {comp_stats.get('zinc_range_avg', 0):.1f}%)")
        if comp_stats.get('magnesium_range'):
            print(f"  Mg: {comp_stats.get('magnesium_range_min', 0):.1f}% - {comp_stats.get('magnesium_range_max', 0):.1f}% (平均: {comp_stats.get('magnesium_range_avg', 0):.1f}%)")
        if comp_stats.get('aluminum_range'):
            print(f"  Al: {comp_stats.get('aluminum_range_min', 0):.1f}% - {comp_stats.get('aluminum_range_max', 0):.1f}% (平均: {comp_stats.get('aluminum_range_avg', 0):.1f}%)")
        if comp_stats.get('copper_range'):
            print(f"  Cu: {comp_stats.get('copper_range_min', 0):.1f}% - {comp_stats.get('copper_range_max', 0):.1f}% (平均: {comp_stats.get('copper_range_avg', 0):.1f}%)")

        print(f"\n力学性能范围:")
        mech_stats = stats.get('mechanical_stats', {})
        if mech_stats.get('uts_range'):
            print(f"  抗拉强度: {mech_stats.get('uts_range_min', 0):.0f} - {mech_stats.get('uts_range_max', 0):.0f} MPa (平均: {mech_stats.get('uts_range_avg', 0):.0f} MPa)")
        if mech_stats.get('ys_range'):
            print(f"  屈服强度: {mech_stats.get('ys_range_min', 0):.0f} - {mech_stats.get('ys_range_max', 0):.0f} MPa (平均: {mech_stats.get('ys_range_avg', 0):.0f} MPa)")
        if mech_stats.get('elongation_range'):
            print(f"  延伸率: {mech_stats.get('elongation_range_min', 0):.1f}% - {mech_stats.get('elongation_range_max', 0):.1f}% (平均: {mech_stats.get('elongation_range_avg', 0):.1f}%)")
        if mech_stats.get('hardness_range'):
            print(f"  硬度: {mech_stats.get('hardness_range_min', 0):.0f} - {mech_stats.get('hardness_range_max', 0):.0f} HV (平均: {mech_stats.get('hardness_range_avg', 0):.0f} HV)")

        print(f"\n工艺方法分布:")
        for method, count in stats.get('processing_methods', {}).items():
            print(f"  {method}: {count} 次")

        print(f"\n主要微观结构相:")
        phases = stats.get('microstructure_phases', {})
        top_phases = sorted(phases.items(), key=lambda x: x[1], reverse=True)[:5]
        for phase, count in top_phases:
            print(f"  {phase}: {count} 次")

if __name__ == "__main__":
    print("\n=== 批量合金信息提取系统 ===")
    print("正在初始化...")

    # 可调整的参数
    MAX_CHUNKS_PER_QUERY = 25  # 每个查询检索的chunk数量，可根据需要调整（建议15-50）
    
    # 自动生成锌合金查询词
    zinc_alloy_systems = [
        "Zn-Mg", "Zn-Al", "Zn-Cu", "Zn-Ag", "Zn-Mn", 
        "Zn-Sn", "Zn-Ni", "Zn-Ca", "Zn-Mg-Ca", "Zn-Al-Mg",
        "Zn-Al-Cu", "Zn-Mg-Al", "Zn-Cu-Ti", "Zn-Mg-Mn",
        "Zn-Ca-Mg", "Zn-Ag-Cu", "Zn-Mn-Mg", "Zn-Sn-Mg"
    ]
    
    # 为每个锌合金体系生成多个查询词
    test_queries = []
    for alloy_system in zinc_alloy_systems:
        test_queries.extend([
            f"{alloy_system} alloy composition mechanical properties",
            f"{alloy_system} alloy tensile strength yield strength",
            f"{alloy_system} alloy heat treatment aging",
            f"{alloy_system} alloy corrosion resistance",
            f"{alloy_system} alloy microstructure phases"
        ])
    
    print(f"\n当前配置: MAX_CHUNKS_PER_QUERY = {MAX_CHUNKS_PER_QUERY}")
    print(f"生成的查询词数量: {len(test_queries)}")
    print(f"如需调整，请修改脚本中的 MAX_CHUNKS_PER_QUERY 参数\n")
    
    extractor = BatchAlloyExtractor(default_max_chunks=MAX_CHUNKS_PER_QUERY)
    print(f"\n开始提取 {len(test_queries)} 个合金查询...")
    
    results = extractor.batch_extract(test_queries, "batch_alloy_data.json")
    
    if results:
        stats = extractor.generate_statistics(results)
        extractor.print_statistics(stats)
        
        print(f"\n=== 提取完成 ===")
        print(f"成功从论文数据库中提取了 {len(results)} 条合金数据")
        print(f"数据已保存到: batch_alloy_data.json")
        print("\n这些数据可用于后续的机器学习预测和多目标优化")
    else:
        print("\n=== 提取完成 ===")
        print("未提取到任何合金数据，请检查数据库内容或调整检索策略")
