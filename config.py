# ─────────────────────────────────────────────
#  config.py
#  All paths, keys, and constants in one place.
#  Never hardcode these anywhere else.
# ─────────────────────────────────────────────

import os

# ── Paths ──────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "data")
ML_DIR           = os.path.join(BASE_DIR, "ml")

DATASET_PATH     = os.path.join(DATA_DIR, "dataset.csv")
JOBS_FILE        = os.path.join(DATA_DIR, "jobs_data.py")
PROFILES_PATH    = os.path.join(DATA_DIR, "ideal_profiles.json")
MODEL_PATH       = os.path.join(ML_DIR, "model.pkl")
TEMP_DIR         = os.path.join(BASE_DIR, "temp")

# ── Groq ───────────────────────────────────────
# Set this as an environment variable — never hardcode your key
# On Windows: setx GROQ_API_KEY "your_key_here"
# On Mac/Linux: export GROQ_API_KEY=your_key_here
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL       = "llama-3.3-70b-versatile"

# ── Model training ─────────────────────────────
TEST_SIZE        = 0.2
RANDOM_STATE     = 42
XGB_PARAMS = {
    "n_estimators":   500,
    "learning_rate":  0.03,
    "max_depth":      6,
    "eval_metric":    "aucpr",
    "random_state":   RANDOM_STATE,
}

# ── Feature columns ────────────────────────────
FEATURE_COLUMNS = [
    "skill_overlap_score",
    "nice_to_have_score",
    "skill_gap_count",
    "domain_match",
    "domain_score",
    "experience_gap",
    "experience_score",
    "resume_skill_count",
    "job_total_required",
    "seniority_fit_encoded",
    "resume_domain_encoded",
    "job_domain_encoded",
    "profile_similarity_score",
]

TARGET_COLUMN    = "selection_label"
WEIGHT_COLUMN    = "match_quality"

# ── Encoding maps ──────────────────────────────
SENIORITY_MAP = {
    "underqualified": 0,
    "fit":            1,
    "overqualified":  2,
}

DOMAIN_MAP = {
    "general":    0,
    "backend":    1,
    "frontend":   2,
    "fullstack":  3,
    "ml":         4,
    "data":       5,
    "devops":     6,
    "cloud":      7,
    "qa":         8,
    "security":   9,
    "mobile":     10,
    "embedded":   11,
    "blockchain": 12,
    "ba":         13,
}

# ── Skill taxonomy ──────────────────────────────
SKILL_TAXONOMY = {
    "python": "python", "python3": "python", "py": "python",
    "django": "django", "flask": "flask", "fastapi": "fastapi",
    "pandas": "pandas", "numpy": "numpy",
    "scikit-learn": "sklearn", "sklearn": "sklearn", "scikit learn": "sklearn",
    "tensorflow": "tensorflow", "tf": "tensorflow", "keras": "keras",
    "pytorch": "pytorch", "torch": "pytorch",
    "java": "java", "j2ee": "java", "java/j2ee": "java",
    "spring": "spring boot", "spring boot": "spring boot", "springboot": "spring boot",
    "javascript": "javascript", "js": "javascript", "es6": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "react": "react", "reactjs": "react", "react.js": "react",
    "react native": "react native",
    "angular": "angular", "angularjs": "angular",
    "vue": "vue", "vuejs": "vue",
    "node": "node", "nodejs": "node", "node.js": "node",
    "express": "express", "expressjs": "express",
    "html": "html", "html5": "html",
    "css": "css", "css3": "css", "sass": "css", "scss": "css",
    "sql": "sql", "mysql": "sql", "postgresql": "sql", "postgres": "sql",
    "sqlite": "sql", "mssql": "sql", "oracle": "sql",
    "mongodb": "mongodb", "mongo": "mongodb",
    "redis": "redis", "firebase": "firebase",
    "aws": "aws", "amazon web services": "aws",
    "microservices": "aws",
    "azure": "azure", "gcp": "gcp", "google cloud": "gcp",
    "docker": "docker", "kubernetes": "kubernetes", "k8s": "kubernetes",
    "terraform": "terraform", "ansible": "ansible", "jenkins": "jenkins",
    "ci/cd": "ci/cd", "cicd": "ci/cd", "github actions": "ci/cd",
    "machine learning": "machine learning", "ml": "machine learning",
    "deep learning": "deep learning", "dl": "deep learning",
    "nlp": "nlp", "natural language processing": "nlp",
    "computer vision": "computer vision",
    "data science": "data science", "data analysis": "data analysis",
    "data modeling": "data modeling", "analytics": "analytics",
    "tableau": "tableau", "power bi": "power bi", "powerbi": "power bi",
    "excel": "excel", "reporting": "reporting",
    "linux": "linux", "unix": "linux", "bash": "linux",
    "networking": "networking", "tcp/ip": "networking",
    "security": "security", "cybersecurity": "security",
    "android": "android", "ios": "ios",
    "swift": "swift", "kotlin": "kotlin", "flutter": "flutter",
    "c": "c", "c++": "c++", "c#": "c#",
    "golang": "golang", "go": "golang",
    "solidity": "solidity", "ethereum": "ethereum",
    "web3": "web3", "blockchain": "blockchain",
    "testing": "testing", "selenium": "selenium",
    "automation": "automation", "pytest": "pytest",
    "git": "git", "github": "git", "gitlab": "git",
    "api": "api", "rest": "api", "restful": "api",
    "graphql": "graphql", "microservices": "microservices",
    "kafka": "kafka", "oop": "oop",
    "agile": "agile", "scrum": "agile",
    "problem solving": "problem solving",
    "data structures": "data structures",
    "integration": "integration", "monitoring": "monitoring",
    "embedded systems": "embedded systems", "firmware": "firmware",
    "iot": "iot", "embedded": "embedded",
}

DOMAIN_KEYWORDS = {
    "ml":         ["machine learning", "deep learning", "tensorflow", "pytorch", "nlp",
                   "computer vision", "sklearn", "model", "neural", "ai", "data science"],
    "data":       ["data analyst", "data engineer", "sql", "tableau", "power bi",
                   "etl", "pipeline", "warehouse", "analytics"],
    "backend":    ["django", "flask", "spring boot", "api", "microservices",
                   "backend", "server", "node", "express", "fastapi"],
    "frontend":   ["react", "angular", "vue", "html", "css", "javascript", "ui",
                   "frontend", "responsive", "redux"],
    "fullstack":  ["fullstack", "full stack", "full-stack", "mern", "mean"],
    "devops":     ["devops", "ci/cd", "docker", "kubernetes", "terraform",
                   "jenkins", "ansible", "pipeline", "infrastructure"],
    "cloud":      ["aws", "azure", "gcp", "cloud", "lambda", "s3", "ec2"],
    "mobile":     ["android", "ios", "swift", "kotlin", "flutter", "react native", "mobile"],
    "security":   ["security", "cybersecurity", "penetration", "firewall",
                   "vulnerability", "owasp", "incident response"],
    "qa":         ["testing", "qa", "quality assurance", "selenium", "automation",
                   "test", "sdet", "cypress"],
    "embedded":   ["embedded", "firmware", "iot", "microcontroller", "rtos", "c", "c++"],
    "blockchain": ["blockchain", "solidity", "ethereum", "web3", "smart contract"],
    "ba":         ["business analyst", "business analysis", "requirements", "excel",
                   "reporting", "stakeholder", "process improvement"],
    "general":    ["software engineer", "software developer", "programmer",
                   "java", "python", "oop", "problem solving"],
}

EXPERIENCE_PATTERNS = [
    r'(\d+)\+?\s*years?\s*of\s*(?:professional\s*)?experience',
    r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:work|industry|relevant)\s*experience',
    r'over\s*(\d+)\s*years?',
    r'(\d+)\+\s*years?',
    r'(\d+)\s*years?\s*experience',
    r'experience\s*(?:of\s*)?(\d+)\s*years?',
]
MAX_EXPERIENCE_CAP = 30

DOMAIN_PARTIAL_CREDIT = {
    ("backend",   "fullstack"):  0.6,
    ("fullstack", "backend"):    0.6,
    ("frontend",  "fullstack"):  0.6,
    ("fullstack", "frontend"):   0.6,
    ("ml",        "data"):       0.7,
    ("data",      "ml"):         0.7,
    ("devops",    "cloud"):      0.6,
    ("cloud",     "devops"):     0.6,
    ("qa",        "general"):    0.4,
    ("general",   "backend"):    0.4,
    ("general",   "frontend"):   0.4,
    ("general",   "fullstack"):  0.4,
    ("general",   "ml"):         0.3,
    ("general",   "data"):       0.3,
    ("qa",        "general"):    0.4,
    ("qa",        "backend"):    0.3,
    ("backend",   "devops"):     0.3,
    ("devops",    "backend"):    0.3,
    ("cloud",     "backend"):    0.4,
    ("cloud",     "fullstack"):  0.4,
    ("backend",   "cloud"):      0.4,
    ("ml",        "backend"):    0.3,
    ("ml",        "fullstack"):  0.3,
    ("data",      "backend"):    0.3,
}