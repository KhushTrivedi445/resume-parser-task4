from sqlalchemy import JSON, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class ResumeRecord(Base):
    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    skills: Mapped[list[str]] = mapped_column(JSON, default=list)
    education: Mapped[list[str]] = mapped_column(JSON, default=list)
    experience: Mapped[list[str]] = mapped_column(JSON, default=list)
