import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retriever.hybrid import HybridRetriever
from app.models.alloy_data import AlloyData, AlloyComposition, ProcessingParameters, Microstructure, MechanicalProperties, CorrosionProperties, TestingConditions
from app.llm.generator import Generator
from typing import Optional, List, Dict, Any

class AlloyInfoExtractor:
    """合金信息提取器"""
    
    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = Generator()
    
    def extract_from_paper(self, paper_title: str, max_chunks: int = 20) -> Optional[AlloyData]:
        """从论文中提取合金信息
        
        Args:
            paper_title: 论文标题
            max_chunks: 检索的最大chunk数量
            
        Returns:
            AlloyData: 提取的合金数据
        """
        # 检索相关chunks
        query = f"关于{paper_title}的合金成分、制备工艺和性能数据"
        retrieval_result = self.retriever.retrieve(query, top_k=max_chunks)
        
        if not retrieval_result or not retrieval_result.chunks:
            print(f"未找到关于{paper_title}的相关信息")
            return None
        
        # 构建上下文 - 从RetrievalResult中提取chunk的text字段
        context_parts = []
        for i, chunk in enumerate(retrieval_result.chunks):
            # RetrievedChunk对象的text字段包含实际内容
            content = chunk.text if hasattr(chunk, 'text') else str(chunk)
            context_parts.append(f"Chunk {i+1}: {content}")
        
        context = "\n".join(context_parts)
        
        # 提取指令
        extraction_prompt = f"""你是一名专业的材料科学研究员，请从以下论文内容中提取锌合金的结构化数据。

论文标题: {paper_title}

论文内容:
{context}

请按照以下JSON格式提取信息，确保数据准确完整：

{{
  "alloy_id": "合金唯一标识符",
  "alloy_system": "合金体系 (如: Zn-Mg, Zn-Al-Mg等)",
  "year": 研究年份,
  "paper_title": "论文标题",
  "doi": "论文DOI",
  "authors": ["作者1", "作者2"],
  "institution": "研究机构",
  "composition": {{
    "zinc": Zn含量 (wt%),
    "magnesium": Mg含量 (wt%),
    "aluminum": Al含量 (wt%),
    "copper": Cu含量 (wt%),
    "silver": Ag含量 (wt%),
    "other_elements": {{"元素": 含量}},
    "impurities": {{"杂质元素": 含量}}
  }},
  "processing": {{
    "melting_temperature": 熔炼温度 (°C),
    "melting_time": 熔炼时间 (min),
    "casting_method": "铸造方法",
    "cooling_rate": 冷却速率 (°C/min),
    "rolling_reduction": 轧制变形量 (%),
    "extrusion_ratio": 挤压比,
    "annealing_temperature": 退火温度 (°C),
    "annealing_time": 退火时间 (h),
    "aging_temperature": 时效温度 (°C),
    "aging_time": 时效时间 (h)
  }},
  "microstructure": {{
    "phases": ["相1", "相2"],
    "grain_size": 平均晶粒尺寸 (μm),
    "precipitates": {{"类型": "析出物类型", "size": "尺寸"}},
    "texture": "织构特征"
  }},
  "mechanical_properties": {{
    "ultimate_tensile_strength": 抗拉强度 (MPa),
    "yield_strength": 屈服强度 (MPa),
    "elongation": 延伸率 (%),
    "hardness": 硬度 (HV),
    "elastic_modulus": 弹性模量 (GPa)
  }},
  "corrosion_properties": {{
    "corrosion_rate": 腐蚀速率 (mm/year),
    "corrosion_potential": 腐蚀电位 (mV),
    "polarization_resistance": 极化电阻 (Ω·cm²)
  }},
  "testing_conditions": {{
    "test_standard": "测试标准",
    "temperature": 测试温度 (°C),
    "humidity": 相对湿度 (%),
    "strain_rate": 应变速率 (s⁻¹),
    "test_equipment": "测试设备"
  }},
  "key_findings": ["关键发现1", "关键发现2"],
  "optimization_goals": ["优化目标1", "优化目标2"],
  "notes": "备注"
}}

如果某些信息在论文中没有提及，请留为null或空列表/字典。
请确保JSON格式正确，不要包含任何额外的解释性文本。
"""
        
        # 调用LLM提取信息
        response = self.generator.generate(question=extraction_prompt, context="")
        
        # 解析JSON响应
        try:
            # 提取JSON部分
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                print(f"无法从响应中提取JSON: {response}")
                return None
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # 转换为AlloyData对象
            alloy_data = AlloyData(**data)
            return alloy_data
            
        except Exception as e:
            print(f"解析提取结果时出错: {e}")
            print(f"响应内容: {response}")
            return None
    
    def batch_extract(self, paper_titles: List[str], output_file: str = "alloy_data.json") -> List[AlloyData]:
        """批量提取多篇论文的合金信息
        
        Args:
            paper_titles: 论文标题列表
            output_file: 输出文件路径
            
        Returns:
            List[AlloyData]: 提取的合金数据列表
        """
        extracted_data = []
        
        for i, title in enumerate(paper_titles, 1):
            print(f"\n=== 提取第 {i}/{len(paper_titles)} 篇: {title} ===")
            try:
                data = self.extract_from_paper(title)
                if data:
                    extracted_data.append(data)
                    print(f"提取成功: {data.alloy_system}")
                else:
                    print(f"提取失败")
            except Exception as e:
                print(f"处理出错: {e}")
        
        # 保存结果
        if extracted_data:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([d.dict() for d in extracted_data], f, ensure_ascii=False, indent=2)
            print(f"\n已保存 {len(extracted_data)} 条合金数据到 {output_file}")
        
        return extracted_data

if __name__ == "__main__":
    extractor = AlloyInfoExtractor()
    
    # 测试提取
    test_papers = [
        "Zn-1.0Mg-0.1Ca合金的组织与性能研究",
        "Al-Zn-Mg-Cu合金时效析出行为研究",
        "锌合金在生物医学领域的应用进展"
    ]
    
    results = extractor.batch_extract(test_papers, "test_alloy_data.json")
    print(f"\n总计提取: {len(results)} 篇论文")