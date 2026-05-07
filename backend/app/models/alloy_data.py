from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AlloyComposition(BaseModel):
    """合金成分"""
    zinc: Optional[float] = Field(None, description="Zn含量 (wt%)")
    magnesium: Optional[float] = Field(None, description="Mg含量 (wt%)")
    aluminum: Optional[float] = Field(None, description="Al含量 (wt%)")
    copper: Optional[float] = Field(None, description="Cu含量 (wt%)")
    silver: Optional[float] = Field(None, description="Ag含量 (wt%)")
    other_elements: Optional[Dict[str, Any]] = Field(default_factory=dict, description="其他元素及其含量")
    impurities: Optional[Dict[str, Any]] = Field(default_factory=dict, description="杂质元素及其含量")
    composition_range: Optional[Dict[str, Dict[str, float]]] = Field(None, description="成分变化范围")
    melting_point: Optional[float] = Field(None, description="合金熔点 (°C)")

class ProcessingParameters(BaseModel):
    """制备工艺参数"""
    melting_temperature: Optional[float] = Field(None, description="熔炼温度 (°C)")
    melting_time: Optional[float] = Field(None, description="熔炼时间 (min)")
    melting_atmosphere: Optional[str] = Field(None, description="熔炼气氛")
    casting_method: Optional[str] = Field(None, description="铸造方法")
    cooling_rate: Optional[float] = Field(None, description="冷却速率 (°C/min)")
    mold_temperature: Optional[float] = Field(None, description="模具温度 (°C)")
    rolling_reduction: Optional[float] = Field(None, description="轧制变形量 (%)")
    extrusion_ratio: Optional[float] = Field(None, description="挤压比")
    deformation_temperature: Optional[float] = Field(None, description="变形温度 (°C)")
    deformation_rate: Optional[float] = Field(None, description="变形速率 (s⁻¹)")
    annealing_temperature: Optional[float] = Field(None, description="退火温度 (°C)")
    annealing_time: Optional[float] = Field(None, description="退火时间 (h)")
    solution_temperature: Optional[float] = Field(None, description="固溶温度 (°C)")
    solution_time: Optional[float] = Field(None, description="固溶时间 (h)")
    aging_temperature: Optional[float] = Field(None, description="时效温度 (°C)")
    aging_time: Optional[float] = Field(None, description="时效时间 (h)")
    surface_treatment: Optional[str] = Field(None, description="表面处理工艺")
    process_route: Optional[List[str]] = Field(default_factory=list, description="工艺路线")

class Microstructure(BaseModel):
    """微观结构"""
    phases: Optional[List[str]] = Field(default_factory=list, description="主要相组成")
    phase_fraction: Optional[Dict[str, float]] = Field(None, description="各相体积分数")
    grain_size: Optional[Any] = Field(None, description="平均晶粒尺寸 (μm)")
    grain_size_distribution: Optional[Dict[str, float]] = Field(None, description="晶粒尺寸分布")
    precipitates: Optional[Any] = Field(None, description="析出物信息")
    precipitate_size: Optional[float] = Field(None, description="析出物尺寸 (nm)")
    precipitate_density: Optional[float] = Field(None, description="析出物密度 (个/μm³)")
    texture: Optional[str] = Field(None, description="织构特征")
    dislocation_density: Optional[float] = Field(None, description="位错密度 (m⁻²)")

class MechanicalProperties(BaseModel):
    """力学性能"""
    ultimate_tensile_strength: Optional[float] = Field(None, description="抗拉强度 (MPa)")
    yield_strength: Optional[float] = Field(None, description="屈服强度 (MPa)")
    elongation: Optional[float] = Field(None, description="延伸率 (%)")
    hardness: Optional[float] = Field(None, description="硬度 (HV)")
    elastic_modulus: Optional[float] = Field(None, description="弹性模量 (GPa)")
    poisson_ratio: Optional[float] = Field(None, description="泊松比")
    fatigue_strength: Optional[float] = Field(None, description="疲劳强度 (MPa)")
    impact_energy: Optional[float] = Field(None, description="冲击吸收功 (J)")
    compressive_strength: Optional[float] = Field(None, description="压缩强度 (MPa)")
    shear_strength: Optional[float] = Field(None, description="剪切强度 (MPa)")
    properties_at_temperature: Optional[Dict[float, Dict[str, float]]] = Field(None, description="不同温度下的性能")

class CorrosionProperties(BaseModel):
    """耐腐蚀性能"""
    corrosion_rate: Optional[float] = Field(None, description="腐蚀速率 (mm/year)")
    corrosion_potential: Optional[float] = Field(None, description="腐蚀电位 (mV)")
    polarization_resistance: Optional[float] = Field(None, description="极化电阻 (Ω·cm²)")
    icorr: Optional[float] = Field(None, description="腐蚀电流密度 (μA/cm²)")
    corrosion_test_method: Optional[str] = Field(None, description="腐蚀测试方法")
    test_medium: Optional[str] = Field(None, description="测试介质")

class TestingConditions(BaseModel):
    """测试条件"""
    test_standard: Optional[str] = Field(None, description="测试标准")
    temperature: Optional[float] = Field(None, description="测试温度 (°C)")
    humidity: Optional[float] = Field(None, description="相对湿度 (%)")
    strain_rate: Optional[float] = Field(None, description="应变速率 (s⁻¹)")
    test_equipment: Optional[str] = Field(None, description="测试设备")
    loading_type: Optional[str] = Field(None, description="加载类型")
    test_duration: Optional[float] = Field(None, description="测试持续时间")

class AlloyData(BaseModel):
    """合金完整数据模型"""
    alloy_id: Optional[str] = Field(None, description="合金唯一标识符")
    alloy_system: Optional[str] = Field(None, description="合金体系 (如: Zn-Mg, Zn-Al-Mg等)")
    year: Optional[int] = Field(None, description="研究年份")
    paper_title: Optional[str] = Field(None, description="论文标题")
    doi: Optional[str] = Field(None, description="论文DOI")
    authors: Optional[List[str]] = Field(default_factory=list, description="作者列表")
    institution: Optional[str] = Field(None, description="研究机构")
    batch_id: Optional[str] = Field(None, description="批次标识")
    
    # 核心数据
    composition: Optional[AlloyComposition] = Field(None, description="合金成分")
    processing: Optional[ProcessingParameters] = Field(None, description="制备工艺")
    microstructure: Optional[Microstructure] = Field(None, description="微观结构")
    mechanical_properties: Optional[MechanicalProperties] = Field(None, description="力学性能")
    corrosion_properties: Optional[CorrosionProperties] = Field(None, description="耐腐蚀性能")
    testing_conditions: Optional[TestingConditions] = Field(None, description="测试条件")
    
    # 附加信息
    key_findings: Optional[List[str]] = Field(default_factory=list, description="关键发现")
    optimization_goals: Optional[List[str]] = Field(default_factory=list, description="优化目标")
    simulation_parameters: Optional[Dict[str, Any]] = Field(None, description="有限元模拟参数")
    experimental_verification: Optional[Dict[str, Any]] = Field(None, description="实验验证结果")
    notes: Optional[str] = Field(None, description="备注")

    class Config:
        json_schema_extra = {
            "example": {
                "alloy_id": "ZnMg0.5_2024_001",
                "alloy_system": "Zn-Mg",
                "year": 2024,
                "paper_title": "Zn-Mg合金的力学性能研究",
                "doi": "10.1000/example",
                "authors": ["Zhang, S", "Li, J"],
                "institution": "University of Science and Technology",
                "composition": {
                    "zinc": 99.5,
                    "magnesium": 0.5,
                    "other_elements": {},
                    "impurities": {"Fe": 0.01, "Pb": 0.005}
                },
                "processing": {
                    "melting_temperature": 450,
                    "casting_method": "sand casting",
                    "annealing_temperature": 200,
                    "annealing_time": 1
                },
                "mechanical_properties": {
                    "ultimate_tensile_strength": 200,
                    "yield_strength": 150,
                    "elongation": 15,
                    "hardness": 60
                }
            }
        }