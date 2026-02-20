from sqlalchemy import create_engine, Column, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

from src.config import DATABASE_URL

Base = declarative_base()


class DailyMonitoring(Base):
    __tablename__ = 'daily_monitoring'
    
    date = Column(DateTime, primary_key=True)
    actual_consumption = Column(Float)
    epias_forecast = Column(Float)
    model_prediction = Column(Float)
    
    def __repr__(self):
        return f"<DailyMonitoring(date={self.date}, actual={self.actual_consumption}, forecast={self.epias_forecast}, prediction={self.model_prediction})>"


class Database:
    def __init__(self, db_url: str = None):
        url = db_url or DATABASE_URL
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def upsert_monitoring_data(self, date_val: datetime, actual=None, forecast=None, prediction=None):
        session = self.Session()
        try:
            record = session.query(DailyMonitoring).filter_by(date=date_val).first()
            if not record:
                record = DailyMonitoring(date=date_val)
                session.add(record)
            
            if actual is not None:
                record.actual_consumption = actual
            if forecast is not None:
                record.epias_forecast = forecast
            if prediction is not None:
                record.model_prediction = prediction
                
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_monitoring_data(self):
        session = self.Session()
        try:
            data = session.query(DailyMonitoring).order_by(DailyMonitoring.date).all()
            return data
        finally:
            session.close()


if __name__ == "__main__":
    db = Database()
    print("Database initialized.")
