import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROJECT_NAME: str = "Resume Parser & Job Matcher"
    API_V1_STR: str = "/api/v1"

    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    HUGGINGFACEHUB_API_TOKEN: str | None = os.getenv(
        "HUGGINGFACEHUB_API_TOKEN"
    )

    # Local embedding model; no hosted embedding API key required.
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Maximum size of uploaded resume PDFs (default: 5 MB).
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 5 * 1024 * 1024))

    # Database URL
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///./data/resumes.db"
    )

    @property
    def database_url(self) -> str:
        return self.DATABASE_URL


settings = Settings()
