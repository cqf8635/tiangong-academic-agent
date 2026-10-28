"""Academic Taxonomy & Sub-category Distribution Engine for TianGong Radar.

Authoritative mapping for arXiv's 8 major disciplines, sub-domains, and 150+ granular subject codes
with full English/Chinese bilingual descriptions and query builder support.
"""

from typing import Dict, List, Any, Optional

ACADEMIC_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "PHYSICS": {
        "name_zh": "物理学全系",
        "name_en": "Physics",
        "icon": "🌌",
        "groups": {
            "astro-ph": {
                "name_zh": "天体物理学",
                "name_en": "Astrophysics",
                "arxiv_prefix": "astro-ph",
                "subcategories": {
                    "astro-ph.CO": {"name_zh": "宇宙学与河外天体物理", "name_en": "Cosmology and Nongalactic Astrophysics"},
                    "astro-ph.EP": {"name_zh": "地球与行星天体物理", "name_en": "Earth and Planetary Astrophysics"},
                    "astro-ph.GA": {"name_zh": "星系天体物理", "name_en": "Astrophysics of Galaxies"},
                    "astro-ph.HE": {"name_zh": "高能天体物理", "name_en": "High Energy Astrophysical Phenomena"},
                    "astro-ph.IM": {"name_zh": "天文仪器与方法", "name_en": "Instrumentation and Methods for Astrophysics"},
                    "astro-ph.SR": {"name_zh": "太阳与恒星天体物理", "name_en": "Solar and Stellar Astrophysics"},
                }
            },
            "cond-mat": {
                "name_zh": "凝聚态物理",
                "name_en": "Condensed Matter",
                "arxiv_prefix": "cond-mat",
                "subcategories": {
                    "cond-mat.dis-nn": {"name_zh": "无序系统与神经网络", "name_en": "Disordered Systems and Neural Networks"},
                    "cond-mat.mes-hall": {"name_zh": "介观与霍尔效应", "name_en": "Mesoscale and Nanoscale Physics"},
                    "cond-mat.mtrl-sci": {"name_zh": "材料科学", "name_en": "Materials Science"},
                    "cond-mat.other": {"name_zh": "其他凝聚态物理", "name_en": "Other Condensed Matter"},
                    "cond-mat.quant-gas": {"name_zh": "量子气体", "name_en": "Quantum Gases"},
                    "cond-mat.soft": {"name_zh": "软凝聚态物理", "name_en": "Soft Condensed Matter"},
                    "cond-mat.stat-mech": {"name_zh": "统计力学", "name_en": "Statistical Mechanics"},
                    "cond-mat.str-el": {"name_zh": "强关联电子系统", "name_en": "Strongly Correlated Electrons"},
                    "cond-mat.supr-con": {"name_zh": "超导电性", "name_en": "Superconductivity"},
                }
            },
            "quant-ph": {
                "name_zh": "量子物理与量子信息",
                "name_en": "Quantum Physics",
                "arxiv_prefix": "quant-ph",
                "subcategories": {
                    "quant-ph": {"name_zh": "量子力学、量子通信与量子计算", "name_en": "Quantum Physics & Quantum Info"},
                }
            },
            "gr-qc": {
                "name_zh": "广义相对论与量子引力",
                "name_en": "General Relativity and Quantum Cosmology",
                "arxiv_prefix": "gr-qc",
                "subcategories": {
                    "gr-qc": {"name_zh": "引力、黑洞与时空物理", "name_en": "General Relativity & Quantum Cosmology"},
                }
            },
            "hep": {
                "name_zh": "高能与粒子物理",
                "name_en": "High Energy Physics",
                "arxiv_prefix": "hep",
                "subcategories": {
                    "hep-th": {"name_zh": "高能物理 - 理论 (场论/弦论)", "name_en": "High Energy Physics - Theory"},
                    "hep-ph": {"name_zh": "高能物理 - 唯象学", "name_en": "High Energy Physics - Phenomenology"},
                    "hep-lat": {"name_zh": "高能物理 - 格点理论", "name_en": "High Energy Physics - Lattice"},
                    "hep-ex": {"name_zh": "高能物理 - 实验", "name_en": "High Energy Physics - Experiment"},
                }
            },
            "nucl": {
                "name_zh": "核物理",
                "name_en": "Nuclear Physics",
                "arxiv_prefix": "nucl",
                "subcategories": {
                    "nucl-th": {"name_zh": "核理论", "name_en": "Nuclear Theory"},
                    "nucl-ex": {"name_zh": "核实验", "name_en": "Nuclear Experiment"},
                }
            },
            "nlin": {
                "name_zh": "非线性科学",
                "name_en": "Nonlinear Sciences",
                "arxiv_prefix": "nlin",
                "subcategories": {
                    "nlin.AO": {"name_zh": "自适应与涌现", "name_en": "Adaptation and Self-Organizing Systems"},
                    "nlin.CD": {"name_zh": "混沌动力学", "name_en": "Chaotic Dynamics"},
                    "nlin.CG": {"name_zh": "元胞自动机", "name_en": "Cellular Automata and Lattice Gases"},
                    "nlin.PS": {"name_zh": "图案形成与孤子", "name_en": "Pattern Formation and Solitons"},
                    "nlin.SI": {"name_zh": "精确可积系统", "name_en": "Exactly Solvable and Integrable Systems"},
                }
            },
            "math-ph": {
                "name_zh": "数理物理",
                "name_en": "Mathematical Physics",
                "arxiv_prefix": "math-ph",
                "subcategories": {
                    "math-ph": {"name_zh": "数学物理与严格推演", "name_en": "Mathematical Physics"},
                }
            },
            "physics": {
                "name_zh": "跨学科与应用物理全系",
                "name_en": "Applied Physics & Interdisciplinary",
                "arxiv_prefix": "physics",
                "subcategories": {
                    "physics.app-ph": {"name_zh": "应用物理学", "name_en": "Applied Physics"},
                    "physics.optics": {"name_zh": "光学与光子学", "name_en": "Optics"},
                    "physics.flu-dyn": {"name_zh": "流体动力学", "name_en": "Fluid Dynamics"},
                    "physics.chem-ph": {"name_zh": "化学物理", "name_en": "Chemical Physics"},
                    "physics.bio-ph": {"name_zh": "生物物理", "name_en": "Biological Physics"},
                    "physics.comp-ph": {"name_zh": "计算物理学", "name_en": "Computational Physics"},
                    "physics.plasm-ph": {"name_zh": "等离子体物理", "name_en": "Plasma Physics"},
                    "physics.soc-ph": {"name_zh": "物理与社会科学 (复杂系统)", "name_en": "Physics and Society"},
                    "physics.acc-ph": {"name_zh": "加速器物理", "name_en": "Accelerator Physics"},
                    "physics.ao-ph": {"name_zh": "大气与海洋物理", "name_en": "Atmospheric and Oceanic Physics"},
                    "physics.atm-clus": {"name_zh": "原子与分子团簇", "name_en": "Atomic and Molecular Clusters"},
                    "physics.atom-ph": {"name_zh": "原子与分子物理", "name_en": "Atomic Physics"},
                    "physics.class-ph": {"name_zh": "经典物理学", "name_en": "Classical Physics"},
                    "physics.data-an": {"name_zh": "数据分析与统计物理", "name_en": "Data Analysis, Statistics and Probability"},
                    "physics.ed-ph": {"name_zh": "物理教育与教学法", "name_en": "Physics Education"},
                    "physics.gen-ph": {"name_zh": "通用基础物理", "name_en": "General Physics"},
                    "physics.geo-ph": {"name_zh": "地球物理学", "name_en": "Geophysics"},
                    "physics.hist-ph": {"name_zh": "物理学史与哲学", "name_en": "History and Philosophy of Physics"},
                    "physics.ins-det": {"name_zh": "仪器与探测技术", "name_en": "Instrumentation and Detectors"},
                    "physics.med-ph": {"name_zh": "医学物理学", "name_en": "Medical Physics"},
                    "physics.pop-ph": {"name_zh": "物理普及与前沿通论", "name_en": "Popular Physics"},
                    "physics.space-ph": {"name_zh": "空间物理学", "name_en": "Space Physics"},
                }
            }
        }
    },
    "CS": {
        "name_zh": "计算机科学",
        "name_en": "Computer Science",
        "icon": "💻",
        "groups": {
            "cs-ai": {
                "name_zh": "人工智能与感知计算",
                "name_en": "AI & Perception",
                "arxiv_prefix": "cs",
                "subcategories": {
                    "cs.AI": {"name_zh": "通用人工智能 (AI)", "name_en": "Artificial Intelligence"},
                    "cs.LG": {"name_zh": "机器学习 (ML)", "name_en": "Machine Learning"},
                    "cs.CL": {"name_zh": "计算语言与大语言模型 (NLP)", "name_en": "Computation and Language"},
                    "cs.CV": {"name_zh": "计算机视觉与模式识别 (CV)", "name_en": "Computer Vision and Pattern Recognition"},
                    "cs.RO": {"name_zh": "机器人学与具身智能", "name_en": "Robotics"},
                    "cs.NE": {"name_zh": "神经与进化计算", "name_en": "Neural and Evolutionary Computing"},
                    "cs.MA": {"name_zh": "多智能体系统 (Multiagent)", "name_en": "Multiagent Systems"},
                    "cs.SD": {"name_zh": "声音与音频处理", "name_en": "Sound and Audio Processing"},
                }
            },
            "cs-sys": {
                "name_zh": "计算机系统、体系结构与网络",
                "name_en": "Systems & Architecture",
                "arxiv_prefix": "cs",
                "subcategories": {
                    "cs.DC": {"name_zh": "分布式计算与云计算", "name_en": "Distributed, Parallel, and Cluster Computing"},
                    "cs.CR": {"name_zh": "密码学与网络安全", "name_en": "Cryptography and Security"},
                    "cs.NI": {"name_zh": "计算机网络与互联网通信", "name_en": "Networking and Internet Architecture"},
                    "cs.OS": {"name_zh": "操作系统", "name_en": "Operating Systems"},
                    "cs.AR": {"name_zh": "硬件与体系结构", "name_en": "Hardware Architecture"},
                    "cs.DB": {"name_zh": "数据库与数据管理", "name_en": "Databases"},
                    "cs.SE": {"name_zh": "软件工程", "name_en": "Software Engineering"},
                    "cs.PL": {"name_zh": "程序设计语言与编译器", "name_en": "Programming Languages"},
                    "cs.PF": {"name_zh": "系统性能与评估", "name_en": "Performance and Benchmarking"},
                    "cs.SY": {"name_zh": "系统与控制架构", "name_en": "Systems and Control"},
                }
            },
            "cs-theory": {
                "name_zh": "理论计算机科学与离散数学",
                "name_en": "Theory & Mathematics",
                "arxiv_prefix": "cs",
                "subcategories": {
                    "cs.DS": {"name_zh": "数据结构与算法", "name_en": "Data Structures and Algorithms"},
                    "cs.CC": {"name_zh": "计算复杂性理论", "name_en": "Computational Complexity"},
                    "cs.LO": {"name_zh": "计算机科学逻辑", "name_en": "Logic in Computer Science"},
                    "cs.DM": {"name_zh": "离散数学", "name_en": "Discrete Mathematics"},
                    "cs.FL": {"name_zh": "形式语言与自动机", "name_en": "Formal Languages and Automata Theory"},
                    "cs.CG": {"name_zh": "计算几何", "name_en": "Computational Geometry"},
                    "cs.GT": {"name_zh": "计算机博弈论与算法经济", "name_en": "Computer Science and Game Theory"},
                    "cs.SC": {"name_zh": "符号计算与计算机代数", "name_en": "Symbolic Computation"},
                    "cs.IT": {"name_zh": "信息论", "name_en": "Information Theory"},
                }
            },
            "cs-applied": {
                "name_zh": "应用交叉、前沿交叉与数字媒体",
                "name_en": "Applied & Interdisciplinary",
                "arxiv_prefix": "cs",
                "subcategories": {
                    "cs.HC": {"name_zh": "人机交互 (HCI)", "name_en": "Human-Computer Interaction"},
                    "cs.IR": {"name_zh": "信息检索与推荐系统", "name_en": "Information Retrieval"},
                    "cs.GR": {"name_zh": "计算机图形学", "name_en": "Graphics"},
                    "cs.MM": {"name_zh": "多媒体系统与编解码", "name_en": "Multimedia"},
                    "cs.SI": {"name_zh": "社交网络与信息网络", "name_en": "Social and Information Networks"},
                    "cs.CY": {"name_zh": "计算机与社会治理", "name_en": "Computers and Society"},
                    "cs.CE": {"name_zh": "计算工程、金融与科学计算", "name_en": "Computational Engineering, Finance, and Science"},
                    "cs.NA": {"name_zh": "科学计算与数值算法", "name_en": "Numerical Analysis"},
                    "cs.MS": {"name_zh": "数学计算软件", "name_en": "Mathematical Software"},
                    "cs.DL": {"name_zh": "数字图书馆与学术情报", "name_en": "Digital Libraries"},
                    "cs.ET": {"name_zh": "新兴计算技术与量子体系", "name_en": "Emerging Technologies"},
                    "cs.GL": {"name_zh": "综合文献与通论", "name_en": "General Literature"},
                    "cs.OH": {"name_zh": "其他计算机科学前沿", "name_en": "Other Computer Science"},
                }
            }
        }
    },
    "MATH": {
        "name_zh": "数学全系",
        "name_en": "Mathematics",
        "icon": "📐",
        "groups": {
            "math-analysis": {
                "name_zh": "分析、微分方程与科学计算",
                "name_en": "Analysis & Differential Equations",
                "arxiv_prefix": "math",
                "subcategories": {
                    "math.AP": {"name_zh": "偏微分方程 (PDE)", "name_en": "Analysis of PDEs"},
                    "math.CA": {"name_zh": "经典分析与常微分方程", "name_en": "Classical Analysis and ODEs"},
                    "math.FA": {"name_zh": "泛函分析", "name_en": "Functional Analysis"},
                    "math.DS": {"name_zh": "动力系统", "name_en": "Dynamical Systems"},
                    "math.CV": {"name_zh": "复变函数与多复变", "name_en": "Complex Variables"},
                    "math.SP": {"name_zh": "谱理论", "name_en": "Spectral Theory"},
                    "math.OA": {"name_zh": "算子代数", "name_en": "Operator Algebras"},
                    "math.NA": {"name_zh": "数值分析与科学计算", "name_en": "Numerical Analysis"},
                }
            },
            "math-algebra": {
                "name_zh": "代数、数论与代数结构",
                "name_en": "Algebra & Number Theory",
                "arxiv_prefix": "math",
                "subcategories": {
                    "math.AG": {"name_zh": "代数几何", "name_en": "Algebraic Geometry"},
                    "math.AC": {"name_zh": "交换代数", "name_en": "Commutative Algebra"},
                    "math.RA": {"name_zh": "环与代数", "name_en": "Rings and Algebras"},
                    "math.GR": {"name_zh": "群论", "name_en": "Group Theory"},
                    "math.RT": {"name_zh": "表示论", "name_en": "Representation Theory"},
                    "math.NT": {"name_zh": "数论", "name_en": "Number Theory"},
                    "math.QA": {"name_zh": "量子代数", "name_en": "Quantum Algebra"},
                    "math.KT": {"name_zh": "K-理论与同调代数", "name_en": "K-Theory and Homology"},
                }
            },
            "math-geom-topo": {
                "name_zh": "几何与拓扑学全系",
                "name_en": "Geometry & Topology",
                "arxiv_prefix": "math",
                "subcategories": {
                    "math.DG": {"name_zh": "微分几何", "name_en": "Differential Geometry"},
                    "math.AT": {"name_zh": "代数拓扑", "name_en": "Algebraic Topology"},
                    "math.GT": {"name_zh": "几何拓扑", "name_en": "Geometric Topology"},
                    "math.MG": {"name_zh": "度量几何", "name_en": "Metric Geometry"},
                    "math.SG": {"name_zh": "辛几何", "name_en": "Symplectic Geometry"},
                    "math.GN": {"name_zh": "一般拓扑学", "name_en": "General Topology"},
                    "math.CT": {"name_zh": "范畴论", "name_en": "Category Theory"},
                    "math.OC": {"name_zh": "运筹学、优化与控制论", "name_en": "Optimization and Control"},
                }
            },
            "math-prob-logic": {
                "name_zh": "概率随机、离散组合与数理逻辑",
                "name_en": "Probability, Logic & Foundations",
                "arxiv_prefix": "math",
                "subcategories": {
                    "math.PR": {"name_zh": "概率论与随机过程", "name_en": "Probability"},
                    "math.CO": {"name_zh": "组合数学与图论", "name_en": "Combinatorics"},
                    "math.LO": {"name_zh": "数理逻辑与模型论", "name_en": "Mathematical Logic"},
                    "math.IT": {"name_zh": "数学信息论", "name_en": "Information Theory"},
                    "math.MP": {"name_zh": "数理物理", "name_en": "Mathematical Physics"},
                    "math.ST": {"name_zh": "数理统计理论", "name_en": "Mathematical Statistics"},
                    "math.GM": {"name_zh": "综合与通论数学", "name_en": "General Mathematics"},
                    "math.HO": {"name_zh": "数学史与概览", "name_en": "History and Overview"},
                }
            }
        }
    },
    "Q-BIO": {
        "name_zh": "定量生物学与生物医药",
        "name_en": "Quantitative Biology",
        "icon": "🧬",
        "groups": {
            "q-bio": {
                "name_zh": "生命科学与医学定量建模",
                "name_en": "Quantitative Biology Subdisciplines",
                "arxiv_prefix": "q-bio",
                "subcategories": {
                    "q-bio.BM": {"name_zh": "生物分子结构与动力学", "name_en": "Biomolecules"},
                    "q-bio.NC": {"name_zh": "神经科学与脑认知", "name_en": "Neurons and Cognition"},
                    "q-bio.GN": {"name_zh": "基因组学与生物信息", "name_en": "Genomics"},
                    "q-bio.MN": {"name_zh": "分子与细胞调控网络", "name_en": "Molecular Networks"},
                    "q-bio.QM": {"name_zh": "生物医学定量方法", "name_en": "Quantitative Methods"},
                    "q-bio.PE": {"name_zh": "种群与生态进化", "name_en": "Populations and Evolution"},
                    "q-bio.CB": {"name_zh": "细胞行为学", "name_en": "Cell Behavior"},
                    "q-bio.SC": {"name_zh": "亚细胞过程与动力学", "name_en": "Subcellular Processes"},
                    "q-bio.TO": {"name_zh": "组织与器官建模", "name_en": "Tissues and Organs"},
                    "q-bio.OT": {"name_zh": "其他定量生物学", "name_en": "Other Quantitative Biology"},
                }
            }
        }
    },
    "Q-FIN": {
        "name_zh": "量化金融与金融工程",
        "name_en": "Quantitative Finance",
        "icon": "📈",
        "groups": {
            "q-fin": {
                "name_zh": "数理金融各领域",
                "name_en": "Quantitative Finance Disciplines",
                "arxiv_prefix": "q-fin",
                "subcategories": {
                    "q-fin.PR": {"name_zh": "衍生品与资产定价", "name_en": "Pricing of Securities"},
                    "q-fin.RM": {"name_zh": "风险管理与控制", "name_en": "Risk Management"},
                    "q-fin.PM": {"name_zh": "资产组合管理", "name_en": "Portfolio Management"},
                    "q-fin.TR": {"name_zh": "交易策略与市场微观结构", "name_en": "Trading and Market Microstructure"},
                    "q-fin.MF": {"name_zh": "数理金融学", "name_en": "Mathematical Finance"},
                    "q-fin.CP": {"name_zh": "计算金融与量化模拟", "name_en": "Computational Finance"},
                    "q-fin.ST": {"name_zh": "统计金融学与高频数据", "name_en": "Statistical Finance"},
                    "q-fin.EC": {"name_zh": "金融经济学与市场结构", "name_en": "Economics and Finance"},
                    "q-fin.GN": {"name_zh": "普通宏观金融与经济", "name_en": "General Finance"},
                }
            }
        }
    },
    "STAT": {
        "name_zh": "统计学全系",
        "name_en": "Statistics",
        "icon": "📊",
        "groups": {
            "stat": {
                "name_zh": "现代统计分支",
                "name_en": "Statistics Disciplines",
                "arxiv_prefix": "stat",
                "subcategories": {
                    "stat.ML": {"name_zh": "机器学习统计基础", "name_en": "Machine Learning"},
                    "stat.TH": {"name_zh": "数理统计理论", "name_en": "Statistics Theory"},
                    "stat.ME": {"name_zh": "统计推断与方法学", "name_en": "Methodology"},
                    "stat.AP": {"name_zh": "应用统计学", "name_en": "Applications"},
                    "stat.CO": {"name_zh": "计算统计与马尔可夫链", "name_en": "Computation"},
                    "stat.OT": {"name_zh": "其他综合统计学", "name_en": "Other Statistics"},
                }
            }
        }
    },
    "EESS": {
        "name_zh": "电气工程与系统科学",
        "name_en": "Electrical Engineering & Systems",
        "icon": "⚡",
        "groups": {
            "eess": {
                "name_zh": "智能系统与信号工程",
                "name_en": "Electrical & Systems Disciplines",
                "arxiv_prefix": "eess",
                "subcategories": {
                    "eess.SP": {"name_zh": "信号处理 (Signal Processing)", "name_en": "Signal Processing"},
                    "eess.SY": {"name_zh": "系统与控制理论 (Systems and Control)", "name_en": "Systems and Control"},
                    "eess.IV": {"name_zh": "图像与视频处理", "name_en": "Image and Video Processing"},
                    "eess.AS": {"name_zh": "音频与语音信号处理", "name_en": "Audio and Speech Processing"},
                }
            }
        }
    },
    "ECON": {
        "name_zh": "经济学与计量经济",
        "name_en": "Economics",
        "icon": "📉",
        "groups": {
            "econ": {
                "name_zh": "现代经济学分支",
                "name_en": "Economics Disciplines",
                "arxiv_prefix": "econ",
                "subcategories": {
                    "econ.EM": {"name_zh": "计量经济学", "name_en": "Econometrics"},
                    "econ.TH": {"name_zh": "微观经济与理论经济", "name_en": "Theoretical Economics"},
                    "econ.GN": {"name_zh": "宏观与综合经济学", "name_en": "General Economics"},
                }
            }
        }
    }
}


def get_all_subcategories_under_major(major_key: str) -> List[str]:
    """获取某个学科大类（如 PHYSICS）下所有的二级/三级细分子学科代码"""
    major = ACADEMIC_TAXONOMY.get(major_key)
    if not major:
        return []
    res = []
    for g in major.get("groups", {}).values():
        res.extend(list(g.get("subcategories", {}).keys()))
    return res


def get_subcategory_meta(sub_code: str) -> Optional[Dict[str, str]]:
    """根据子学科代码 (如 astro-ph.HE) 查找其中英文名与父级"""
    for major_k, major_v in ACADEMIC_TAXONOMY.items():
        for group_k, group_v in major_v.get("groups", {}).items():
            if sub_code in group_v.get("subcategories", {}):
                meta = group_v["subcategories"][sub_code]
                return {
                    "code": sub_code,
                    "name_zh": meta["name_zh"],
                    "name_en": meta["name_en"],
                    "group_name_zh": group_v["name_zh"],
                    "major_key": major_k,
                    "major_name_zh": major_v["name_zh"],
                }
    return None


def build_arxiv_query_expression(
    query: Optional[str] = None,
    major: Optional[str] = None,
    sub_categories: Optional[List[str]] = None,
    search_in: str = "all",  # all | title | authors | abstract
) -> str:
    """构建精确的 arXiv API 检索表达式，彻底解决子学科穿透与混杂问题。

    例如：
    - 若指定 sub_categories = ['astro-ph.HE', 'astro-ph.CO']，
      严格生成：(cat:astro-ph.HE OR cat:astro-ph.CO)
    - 若还带 query 'pulsar'，search_in = 'title'：
      严格生成：ti:"pulsar" AND (cat:astro-ph.HE OR cat:astro-ph.CO)
    """
    parts = []

    # 1. 文本搜索词约束
    if query and query.strip():
        q_clean = query.strip().replace('"', '')
        prefix_map = {
            "title": "ti",
            "authors": "au",
            "abstract": "abs",
            "all": "all",
        }
        p = prefix_map.get(search_in, "all")
        parts.append(f'{p}:"{q_clean}"')

    # 2. 学科分类约束
    cat_clauses = []
    if sub_categories and len(sub_categories) > 0:
        # 用户显式勾选或点击了特定子学科（如 astro-ph.HE 或 astro-ph 全组）
        for sub in sub_categories:
            sub = sub.strip()
            if not sub:
                continue
            if "." in sub or "-" in sub or sub.startswith("math") or sub.startswith("cs") or sub.startswith("quant-ph"):
                cat_clauses.append(f"cat:{sub}")
            else:
                cat_clauses.append(f"cat:{sub}*")
    elif major and major != "ALL" and major in ACADEMIC_TAXONOMY:
        # 用户只选了大类（如 CS 或 PHYSICS），按该大类下所有的前缀/代码组合
        group_prefixes = set()
        for g in ACADEMIC_TAXONOMY[major].get("groups", {}).values():
            pref = g.get("arxiv_prefix")
            if pref:
                group_prefixes.add(pref)
        for pref in group_prefixes:
            cat_clauses.append(f"cat:{pref}*")
    elif (major == "ALL" or not major) and not parts:
        # 全学科宏观前沿流 (涵盖计算机、物理、数学、生物、电气工程、统计、金融、经济各大类)
        cat_clauses = [
            "cat:cs*",
            "cat:physics*",
            "cat:astro-ph*",
            "cat:cond-mat*",
            "cat:quant-ph*",
            "cat:math*",
            "cat:stat*",
            "cat:eess*",
            "cat:q-bio*",
            "cat:q-fin*",
            "cat:econ*",
        ]

    if cat_clauses:
        if len(cat_clauses) == 1:
            parts.append(cat_clauses[0])
        else:
            joined_cats = " OR ".join(cat_clauses)
            parts.append(f"({joined_cats})")

    if not parts:
        return "(cat:cs* OR cat:physics* OR cat:math* OR cat:stat* OR cat:eess* OR cat:q-bio* OR cat:q-fin* OR cat:econ*)"

    return " AND ".join(parts)
