from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Create virtual environment, install dependencies and run the code:
# 1. Create: python3 -m venv orm_venv
# 2. Activate: source orm_venv/bin/activate
# 3. Install: pip install sqlalchemy==2.0.35
# 4. Run the code: python code_snippets/03_orm.py

if __name__ == "__main__":
    Base = declarative_base()

    # Define  a class that maps to the users table.
    class User(Base):
        __tablename__ = "users"

        id = Column(Integer, primary_key=True)
        name = Column(String)

    # Create an SQLite database in memory.
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    # Create a session used to interact with the database.
    Session = sessionmaker(bind=engine)
    session = Session()

    # Add a new user.
    new_user = User(name="Alice")
    session.add(new_user)
    session.commit()

    # Query the database.
    user = session.query(User).first()
    if user:
        print(f"User ID: {user.id}")  # noqa
        print(f"User name: {user.name}")  # noqa
