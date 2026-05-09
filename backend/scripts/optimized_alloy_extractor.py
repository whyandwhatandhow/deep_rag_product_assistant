import io
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.llm.generator import Generator
from app.models.alloy_data import AlloyData
from app.retriever.hybrid import HybridRetriever


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')



class OptimizedAlloyExtractor:
    """优化后的合金信息提取器"""

    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = Generator()

    def extract_with_strategy(self, query: str, max_chunks: int = 10) -> Optional[AlloyData]:
        """使用优化策略提取合金信息

        关键改进：
        1. 使用英文检索（数据库是英文内容）
        2. 检索具体的合金成分或性能指标
        3. 多次检索增加召回率
        """
        retrieval_queries = [
            f"{query} alloy composition mechanical properties",
            f"{query} tensile strength yield strength microstructure",
            f"{query} heat treatment aging process parameters"
        ]

        all_chunks = []
        seen_texts = set()

        for q in retrieval_queries:
            result = self.retriever.retrieve(q, top_k=max_chunks)
            for chunk in result.chunks:
                if chunk.text not in seen_texts:
                    seen_texts.add(chunk.text)
                    all_chunks.append(chunk)

        if not all_chunks:
            print(f"未找到关于 {query} 的相关信息")
            return None

        # 构建上下文
        context_parts = []
        for i, chunk in enumerate(all_chunks[:10]):
            context_parts.append(f"[Section {i+1}]\n{chunk.text}")
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
  "composition": {{
    "zinc": Zn content in wt%,
    "magnesium": Mg content in wt% (if applicable),
    "aluminum": Al content in wt% (if applicable),
    "other_elements": {{"element": "content"}} (if applicable)
  }},
  "processing": {{
    "melting_temperature": temperature in °C,
    "casting_method": "method used",
    "annealing_temperature": temperature in °C,
    "aging_temperature": temperature in °C,
    "aging_time": time in hours,
    "other_parameters": "any other relevant parameters"
  }},
  "mechanical_properties": {{
    "ultimate_tensile_strength": UTS in MPa,
    "yield_strength": yield strength in MPa,
    "elongation": elongation in %,
    "hardness": hardness in HV
  }},
  "microstructure": {{
    "phases": ["phase1", "phase2"],
    "grain_size": size in μm if mentioned
  }},
  "key_findings": ["main finding 1", "main finding 2"]
}}

If a field cannot be determined from the content, use null. Return ONLY JSON."""

        response = self.generator.generate(question=extraction_prompt, context="")

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
            print(f"响应内容: {response}")
            return None

    def batch_extract(self, queries: List[str], output_file: str = "optimized_alloy_data.json") -> List[AlloyData]:
        """批量提取多个合金的信息"""
        extracted_data = []

        for i, query in enumerate(queries, 1):
            print(f"\n=== 提取 {i}/{len(queries)}: {query} ===")
            try:
                data = self.extract_with_strategy(query)
                if data:
                    extracted_data.append(data)
                    print(f"提取成功: {data.alloy_system}")
                    if data.composition.zinc:
                        zn = data.composition.zinc
                        mg = data.composition.magnesium or 0
                        al = data.composition.aluminum or 0
                        print(f"成分: Zn-{zn:.1f}%, Mg-{mg:.1f}%, Al-{al:.1f}%")
                    if data.mechanical_properties.ultimate_tensile_strength:
                        print(f"抗拉强度: {data.mechanical_properties.ultimate_tensile_strength} MPa")
                else:
                    print(f"提取失败")
            except Exception as e:
                print(f"处理出错: {e}")

        if extracted_data:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([d.model_dump() for d in extracted_data], f, ensure_ascii=False, indent=2)
            print(f"\n已保存 {len(extracted_data)} 条数据到 {output_file}")

        return extracted_data

if __name__ == "__main__":
    print("\n=== 优化后的合金信息提取测试 ===\n")

    # 使用更有效的检索查询
    test_queries = [
        "Zn-Mg alloy hardness tensile strength",
        "Al-Zn-Mg-Cu alloy aging precipitation",
        "zinc alloy corrosion resistance microstructure",
        "Zn-Al alloy mechanical properties erosion",
        "magnesium alloy texture grain size"
    ]

    extractor = OptimizedAlloyExtractor()
    results = extractor.batch_extract(test_queries, "optimized_alloy_data.json")

    print(f"\n\n=== 提取结果统计 ===")
    print(f"成功提取: {len(results)} 条")

    if results:
        # 统计
        from collections import defaultdict
        systems = defaultdict(int)
        total_uts = []
        total_zn = []

        for data in results:
            if data:
                if data.alloy_system:
                    systems[data.alloy_system] += 1
                if data.mechanical_properties and data.mechanical_properties.ultimate_tensile_strength:
                    uts = data.mechanical_properties.ultimate_tensile_strength
                    if isinstance(uts, (int, float)):
                        total_uts.append(uts)
                if data.composition and data.composition.zinc:
                    zn = data.composition.zinc
                    if isinstance(zn, (int, float)):
                        total_zn.append(zn)

        print(f"\n合金体系分布:")
        for sys, count in systems.items():
            print(f"  {sys}: {count} 篇")

        if total_uts:
            print(f"\n抗拉强度范围: {min(total_uts):.0f} - {max(total_uts):.0f} MPa")
            print(f"平均抗拉强度: {sum(total_uts)/len(total_uts):.0f} MPa")

        if total_zn:
            print(f"\nZn含量范围: {min(total_zn):.1f}% - {max(total_zn):.1f}%")