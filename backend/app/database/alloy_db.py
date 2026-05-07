import sqlite3
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import json
from pathlib import Path

class AlloyDatabase:
    """合金专用数据库管理器"""

    def __init__(self, db_path: str = "alloy_database.db"):
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """初始化数据库连接"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """创建数据库表结构"""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alloys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alloy_id TEXT UNIQUE NOT NULL,
                alloy_system TEXT,
                year INTEGER,
                paper_title TEXT,
                doi TEXT,
                authors TEXT,
                institution TEXT,
                batch_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compositions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alloy_id TEXT NOT NULL,
                zinc REAL,
                magnesium REAL,
                aluminum REAL,
                copper REAL,
                silver REAL,
                other_elements TEXT,
                impurities TEXT,
                composition_range TEXT,
                FOREIGN KEY (alloy_id) REFERENCES alloys(alloy_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alloy_id TEXT NOT NULL,
                melting_temperature REAL,
                melting_time REAL,
                melting_atmosphere TEXT,
                casting_method TEXT,
                cooling_rate REAL,
                mold_temperature REAL,
                rolling_reduction REAL,
                extrusion_ratio REAL,
                deformation_temperature REAL,
                deformation_rate REAL,
                annealing_temperature REAL,
                annealing_time REAL,
                solution_temperature REAL,
                solution_time REAL,
                aging_temperature REAL,
                aging_time REAL,
                surface_treatment TEXT,
                process_route TEXT,
                FOREIGN KEY (alloy_id) REFERENCES alloys(alloy_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS microstructure (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alloy_id TEXT NOT NULL,
                phases TEXT,
                phase_fraction TEXT,
                grain_size TEXT,
                grain_size_distribution TEXT,
                precipitates TEXT,
                precipitate_size REAL,
                precipitate_density REAL,
                texture TEXT,
                dislocation_density REAL,
                FOREIGN KEY (alloy_id) REFERENCES alloys(alloy_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mechanical_properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alloy_id TEXT NOT NULL,
                ultimate_tensile_strength REAL,
                yield_strength REAL,
                elongation REAL,
                hardness REAL,
                elastic_modulus REAL,
                poisson_ratio REAL,
                fatigue_strength REAL,
                impact_energy REAL,
                compressive_strength REAL,
                shear_strength REAL,
                properties_at_temperature TEXT,
                FOREIGN KEY (alloy_id) REFERENCES alloys(alloy_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corrosion_properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alloy_id TEXT NOT NULL,
                corrosion_rate REAL,
                corrosion_potential REAL,
                polarization_resistance REAL,
                icorr REAL,
                corrosion_test_method TEXT,
                test_medium TEXT,
                FOREIGN KEY (alloy_id) REFERENCES alloys(alloy_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS testing_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alloy_id TEXT NOT NULL,
                test_standard TEXT,
                temperature REAL,
                humidity REAL,
                strain_rate REAL,
                test_equipment TEXT,
                loading_type TEXT,
                test_duration REAL,
                FOREIGN KEY (alloy_id) REFERENCES alloys(alloy_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS additional_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alloy_id TEXT NOT NULL,
                key_findings TEXT,
                optimization_goals TEXT,
                simulation_parameters TEXT,
                experimental_verification TEXT,
                notes TEXT,
                FOREIGN KEY (alloy_id) REFERENCES alloys(alloy_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alloys_system ON alloys(alloy_system)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alloys_year ON alloys(year)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_compositions_zinc ON compositions(zinc)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mechanical_uts ON mechanical_properties(ultimate_tensile_strength)
        """)

        self.conn.commit()

    def insert_alloy(self, alloy_data: Dict[str, Any]) -> bool:
        """插入合金数据"""
        try:
            cursor = self.conn.cursor()

            alloy_id = alloy_data.get('alloy_id')
            if not alloy_id:
                return False

            cursor.execute("""
                INSERT OR REPLACE INTO alloys 
                (alloy_id, alloy_system, year, paper_title, doi, authors, institution, batch_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                alloy_id,
                alloy_data.get('alloy_system'),
                alloy_data.get('year'),
                alloy_data.get('paper_title'),
                alloy_data.get('doi'),
                json.dumps(alloy_data.get('authors', []), ensure_ascii=False) if alloy_data.get('authors') else None,
                alloy_data.get('institution'),
                alloy_data.get('batch_id')
            ))

            if alloy_data.get('composition'):
                comp = alloy_data['composition']
                cursor.execute("""
                    INSERT OR REPLACE INTO compositions
                    (alloy_id, zinc, magnesium, aluminum, copper, silver, other_elements, impurities, composition_range)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alloy_id,
                    comp.get('zinc'),
                    comp.get('magnesium'),
                    comp.get('aluminum'),
                    comp.get('copper'),
                    comp.get('silver'),
                    json.dumps(comp.get('other_elements', {}), ensure_ascii=False) if comp.get('other_elements') else None,
                    json.dumps(comp.get('impurities', {}), ensure_ascii=False) if comp.get('impurities') else None,
                    json.dumps(comp.get('composition_range', {}), ensure_ascii=False) if comp.get('composition_range') else None
                ))

            if alloy_data.get('processing'):
                proc = alloy_data['processing']
                cursor.execute("""
                    INSERT OR REPLACE INTO processing
                    (alloy_id, melting_temperature, melting_time, melting_atmosphere, casting_method,
                     cooling_rate, mold_temperature, rolling_reduction, extrusion_ratio,
                     deformation_temperature, deformation_rate, annealing_temperature, annealing_time,
                     solution_temperature, solution_time, aging_temperature, aging_time,
                     surface_treatment, process_route)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alloy_id,
                    proc.get('melting_temperature'),
                    proc.get('melting_time'),
                    proc.get('melting_atmosphere'),
                    proc.get('casting_method'),
                    proc.get('cooling_rate'),
                    proc.get('mold_temperature'),
                    proc.get('rolling_reduction'),
                    proc.get('extrusion_ratio'),
                    proc.get('deformation_temperature'),
                    proc.get('deformation_rate'),
                    proc.get('annealing_temperature'),
                    proc.get('annealing_time'),
                    proc.get('solution_temperature'),
                    proc.get('solution_time'),
                    proc.get('aging_temperature'),
                    proc.get('aging_time'),
                    proc.get('surface_treatment'),
                    json.dumps(proc.get('process_route', []), ensure_ascii=False) if proc.get('process_route') else None
                ))

            if alloy_data.get('microstructure'):
                micro = alloy_data['microstructure']
                cursor.execute("""
                    INSERT OR REPLACE INTO microstructure
                    (alloy_id, phases, phase_fraction, grain_size, grain_size_distribution,
                     precipitates, precipitate_size, precipitate_density, texture, dislocation_density)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alloy_id,
                    json.dumps(micro.get('phases', []), ensure_ascii=False) if micro.get('phases') else None,
                    json.dumps(micro.get('phase_fraction', {}), ensure_ascii=False) if micro.get('phase_fraction') else None,
                    str(micro.get('grain_size')) if micro.get('grain_size') else None,
                    json.dumps(micro.get('grain_size_distribution', {}), ensure_ascii=False) if micro.get('grain_size_distribution') else None,
                    str(micro.get('precipitates')) if micro.get('precipitates') else None,
                    micro.get('precipitate_size'),
                    micro.get('precipitate_density'),
                    micro.get('texture'),
                    micro.get('dislocation_density')
                ))

            if alloy_data.get('mechanical_properties'):
                mech = alloy_data['mechanical_properties']
                cursor.execute("""
                    INSERT OR REPLACE INTO mechanical_properties
                    (alloy_id, ultimate_tensile_strength, yield_strength, elongation, hardness,
                     elastic_modulus, poisson_ratio, fatigue_strength, impact_energy,
                     compressive_strength, shear_strength, properties_at_temperature)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alloy_id,
                    mech.get('ultimate_tensile_strength'),
                    mech.get('yield_strength'),
                    mech.get('elongation'),
                    mech.get('hardness'),
                    mech.get('elastic_modulus'),
                    mech.get('poisson_ratio'),
                    mech.get('fatigue_strength'),
                    mech.get('impact_energy'),
                    mech.get('compressive_strength'),
                    mech.get('shear_strength'),
                    json.dumps(mech.get('properties_at_temperature', {}), ensure_ascii=False) if mech.get('properties_at_temperature') else None
                ))

            if alloy_data.get('corrosion_properties'):
                corr = alloy_data['corrosion_properties']
                cursor.execute("""
                    INSERT OR REPLACE INTO corrosion_properties
                    (alloy_id, corrosion_rate, corrosion_potential, polarization_resistance,
                     icorr, corrosion_test_method, test_medium)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    alloy_id,
                    corr.get('corrosion_rate'),
                    corr.get('corrosion_potential'),
                    corr.get('polarization_resistance'),
                    corr.get('icorr'),
                    corr.get('corrosion_test_method'),
                    corr.get('test_medium')
                ))

            if alloy_data.get('testing_conditions'):
                test = alloy_data['testing_conditions']
                cursor.execute("""
                    INSERT OR REPLACE INTO testing_conditions
                    (alloy_id, test_standard, temperature, humidity, strain_rate,
                     test_equipment, loading_type, test_duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alloy_id,
                    test.get('test_standard'),
                    test.get('temperature'),
                    test.get('humidity'),
                    test.get('strain_rate'),
                    test.get('test_equipment'),
                    test.get('loading_type'),
                    test.get('test_duration')
                ))

            if any([alloy_data.get('key_findings'), alloy_data.get('optimization_goals'),
                    alloy_data.get('simulation_parameters'), alloy_data.get('notes')]):
                cursor.execute("""
                    INSERT OR REPLACE INTO additional_info
                    (alloy_id, key_findings, optimization_goals, simulation_parameters,
                     experimental_verification, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    alloy_id,
                    json.dumps(alloy_data.get('key_findings', []), ensure_ascii=False) if alloy_data.get('key_findings') else None,
                    json.dumps(alloy_data.get('optimization_goals', []), ensure_ascii=False) if alloy_data.get('optimization_goals') else None,
                    json.dumps(alloy_data.get('simulation_parameters', {}), ensure_ascii=False) if alloy_data.get('simulation_parameters') else None,
                    json.dumps(alloy_data.get('experimental_verification', {}), ensure_ascii=False) if alloy_data.get('experimental_verification') else None,
                    alloy_data.get('notes')
                ))

            self.conn.commit()
            return True

        except Exception as e:
            print(f"插入数据出错: {e}")
            self.conn.rollback()
            return False

    def batch_import(self, data_list: List[Dict[str, Any]]) -> Tuple[int, int]:
        """批量导入数据
        
        Returns:
            (成功数量, 失败数量)
        """
        success_count = 0
        fail_count = 0

        for data in data_list:
            if self.insert_alloy(data):
                success_count += 1
            else:
                fail_count += 1

        return success_count, fail_count

    def query_alloys(self,
                    alloy_system: Optional[str] = None,
                    zinc_range: Optional[Tuple[float, float]] = None,
                    magnesium_range: Optional[Tuple[float, float]] = None,
                    uts_min: Optional[float] = None,
                    uts_max: Optional[float] = None,
                    ys_min: Optional[float] = None,
                    ys_max: Optional[float] = None,
                    limit: int = 100) -> List[Dict[str, Any]]:
        """高级查询功能"""
        cursor = self.conn.cursor()

        query = """
            SELECT DISTINCT a.*,
                   c.zinc, c.magnesium, c.aluminum, c.copper, c.silver,
                   m.ultimate_tensile_strength, m.yield_strength, m.elongation, m.hardness,
                   p.casting_method, p.aging_temperature, p.aging_time
            FROM alloys a
            LEFT JOIN compositions c ON a.alloy_id = c.alloy_id
            LEFT JOIN mechanical_properties m ON a.alloy_id = m.alloy_id
            LEFT JOIN processing p ON a.alloy_id = p.alloy_id
            WHERE 1=1
        """
        params = []

        if alloy_system:
            query += " AND a.alloy_system LIKE ?"
            params.append(f"%{alloy_system}%")

        if zinc_range:
            query += " AND c.zinc BETWEEN ? AND ?"
            params.extend(zinc_range)

        if magnesium_range:
            query += " AND c.magnesium BETWEEN ? AND ?"
            params.extend(magnesium_range)

        if uts_min:
            query += " AND m.ultimate_tensile_strength >= ?"
            params.append(uts_min)

        if uts_max:
            query += " AND m.ultimate_tensile_strength <= ?"
            params.append(uts_max)

        if ys_min:
            query += " AND m.yield_strength >= ?"
            params.append(ys_min)

        if ys_max:
            query += " AND m.yield_strength <= ?"
            params.append(ys_max)

        query += f" ORDER BY a.created_at DESC LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(dict(row))

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        cursor = self.conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) as count FROM alloys")
        stats['total_alloys'] = cursor.fetchone()['count']

        cursor.execute("SELECT alloy_system, COUNT(*) as count FROM alloys GROUP BY alloy_system")
        stats['by_system'] = {row['alloy_system']: row['count'] for row in cursor.fetchall() if row['alloy_system']}

        cursor.execute("SELECT MIN(zinc) as min, MAX(zinc) as max, AVG(zinc) as avg FROM compositions WHERE zinc IS NOT NULL")
        row = cursor.fetchone()
        stats['zinc_range'] = {'min': row['min'], 'max': row['max'], 'avg': row['avg']}

        cursor.execute("SELECT MIN(magnesium) as min, MAX(magnesium) as max, AVG(magnesium) as avg FROM compositions WHERE magnesium IS NOT NULL")
        row = cursor.fetchone()
        stats['magnesium_range'] = {'min': row['min'], 'max': row['max'], 'avg': row['avg']}

        cursor.execute("""
            SELECT MIN(ultimate_tensile_strength) as min, MAX(ultimate_tensile_strength) as max, 
                   AVG(ultimate_tensile_strength) as avg
            FROM mechanical_properties WHERE ultimate_tensile_strength IS NOT NULL
        """)
        row = cursor.fetchone()
        stats['uts_range'] = {'min': row['min'], 'max': row['max'], 'avg': row['avg']}

        cursor.execute("""
            SELECT MIN(yield_strength) as min, MAX(yield_strength) as max, AVG(yield_strength) as avg
            FROM mechanical_properties WHERE yield_strength IS NOT NULL
        """)
        row = cursor.fetchone()
        stats['ys_range'] = {'min': row['min'], 'max': row['max'], 'avg': row['avg']}

        return stats

    def export_to_csv(self, output_file: str = "alloy_data_export.csv") -> bool:
        """导出数据为CSV格式（适合ML训练）"""
        try:
            cursor = self.conn.cursor()

            query = """
                SELECT 
                    a.alloy_id,
                    a.alloy_system,
                    a.year,
                    c.zinc,
                    c.magnesium,
                    c.aluminum,
                    c.copper,
                    c.silver,
                    m.ultimate_tensile_strength,
                    m.yield_strength,
                    m.elongation,
                    m.hardness,
                    m.elastic_modulus,
                    p.casting_method,
                    p.annealing_temperature,
                    p.aging_temperature,
                    p.aging_time
                FROM alloys a
                LEFT JOIN compositions c ON a.alloy_id = c.alloy_id
                LEFT JOIN mechanical_properties m ON a.alloy_id = m.alloy_id
                LEFT JOIN processing p ON a.alloy_id = p.alloy_id
            """

            cursor.execute(query)
            rows = cursor.fetchall()

            if not rows:
                print("没有数据可导出")
                return False

            headers = [description[0] for description in cursor.description]

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(','.join(headers) + '\n')
                for row in rows:
                    values = [str(val) if val is not None else '' for val in row]
                    f.write(','.join(values) + '\n')

            print(f"数据已导出到 {output_file}")
            return True

        except Exception as e:
            print(f"导出失败: {e}")
            return False

    def export_to_json(self, output_file: str = "alloy_data_export.json") -> bool:
        """导出数据为JSON格式"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT alloy_id FROM alloys")
            alloy_ids = [row['alloy_id'] for row in cursor.fetchall()]

            export_data = []
            for alloy_id in alloy_ids:
                alloy = self.get_alloy_by_id(alloy_id)
                if alloy:
                    export_data.append(alloy)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            print(f"数据已导出到 {output_file}")
            return True

        except Exception as e:
            print(f"导出失败: {e}")
            return False

    def get_alloy_by_id(self, alloy_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取完整合金数据"""
        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM alloys WHERE alloy_id = ?", (alloy_id,))
        row = cursor.fetchone()
        if not row:
            return None

        data = dict(row)

        for table, prefix in [('compositions', 'comp'), ('mechanical_properties', 'mech'),
                              ('processing', 'proc'), ('microstructure', 'micro'),
                              ('corrosion_properties', 'corr'), ('testing_conditions', 'test'),
                              ('additional_info', 'add')]:
            cursor.execute(f"SELECT * FROM {table} WHERE alloy_id = ?", (alloy_id,))
            row = cursor.fetchone()
            if row:
                data.update({k: v for k, v in dict(row).items() if k != 'alloy_id'})

        return data

    def delete_alloy(self, alloy_id: str) -> bool:
        """删除合金数据"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM alloys WHERE alloy_id = ?", (alloy_id,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"删除失败: {e}")
            return False

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
