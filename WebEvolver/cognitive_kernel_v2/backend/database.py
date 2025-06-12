from sqlalchemy import Column, Integer, String, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc
from sqlalchemy import func
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, MO
import json
import os

Base = declarative_base()

POSTGRES_HOST = "postgres"
POSTGRES_PORT = "5432"
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB_USER = "cognitive_kernel_user"
POSTGRES_DB_HISTORY = "cognitive_kernel_history"
POSTGRES_DB_RAW_DATA = "cognitive_kernel_raw_data"
POSTGRES_DB_ANNOTATION = "cognitive_kernel_annotation"

user_db_connector = create_engine(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB_USER}"
)
history_db_connector = create_engine(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB_HISTORY}"
)
raw_data_db_connector = create_engine(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB_RAW_DATA}"
)
annotation_db_connector = create_engine(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB_ANNOTATION}"
)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String)
    password = Column(String)

    def __repr__(self):
        return f"<User(name='{self.username}')>"


class MessageHistory(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String)
    model_name = Column(String)
    username = Column(String)
    initial_message = Column(String)
    messages = Column(String)
    created_time = Column(String)
    updated_time = Column(String)
    archived = Column(Boolean, default=False)

    def __repr__(self):
        return f"<MessageHistory(session_id='{self.session_id}', username='{self.username}')>"


class RawData(Base):
    __tablename__ = "annotations_by_session"

    id = Column(Integer, primary_key=True)
    session_id = Column(String)
    message_id = Column(String)
    username = Column(String)
    raw_data = Column(String)
    created_time = Column(String)
    updated_time = Column(String)

    def __repr__(self):
        return f"<RawData(session_id='{self.session_id}', username='{self.username}')>"


class Annotation(Base):
    __tablename__ = "annotations_by_message"

    id = Column(Integer, primary_key=True)
    session_id = Column(String)
    message_id = Column(String)
    username = Column(String)
    tag = Column(String)
    for_evaluation = Column(Boolean)
    old_message = Column(String)
    suggestion = Column(String)
    annotations = Column(String)
    created_time = Column(String)
    updated_time = Column(String)

    def __repr__(self):
        return (
            f"<Annotation(session_id='{self.session_id}', username='{self.username}')>"
        )


User.metadata.create_all(user_db_connector)
MessageHistory.metadata.create_all(history_db_connector)
RawData.metadata.create_all(raw_data_db_connector)
Annotation.metadata.create_all(annotation_db_connector)
UserSessionLocal = sessionmaker(bind=user_db_connector)
MessageHistorySessionLocal = sessionmaker(bind=history_db_connector)
RawDataSessionLocal = sessionmaker(bind=raw_data_db_connector)
AnnotationSessionLocal = sessionmaker(bind=annotation_db_connector)


def update_or_create_session(session_id, username, model_name, messages, updated_time):
    # Create a new session to interact with the database
    session = MessageHistorySessionLocal()

    print("session_id: ", session_id)
    print("username: ", username)
    print("model_name: ", model_name)
    print("messages: ", messages)
    print("updated_time: ", updated_time)

    # Attempt to find an existing session with the same session_id and username
    existing_session = (
        session.query(MessageHistory)
        .filter_by(session_id=session_id, username=username)
        .first()
    )

    if existing_session:
        # If the session exists, update it
        existing_session.messages = json.dumps(messages)
        existing_session.updated_time = updated_time
    else:
        # If the session does not exist, create a new one
        new_session = MessageHistory(
            session_id=session_id,
            model_name=model_name,
            username=username,
            initial_message=messages[0]["message"][0]["content"],
            messages=json.dumps(messages),
            created_time=updated_time,
            updated_time=updated_time,
            archived=False,
        )
        session.add(new_session)

    # Commit the changes to the database
    session.commit()
    session.close()
    print("Session updated or created successfully.")


def get_sessions_by_username(model_name: str, username: str):
    session = MessageHistorySessionLocal()
    try:
        sessions = (
            session.query(MessageHistory)
            .filter_by(username=username, model_name=model_name, archived=False)
            .order_by(desc(MessageHistory.updated_time))
            .limit(50)
            .with_entities(
                MessageHistory.id,
                MessageHistory.session_id,
                MessageHistory.initial_message,
                MessageHistory.updated_time,
            )
            .all()
        )
        sessions_info = [
            {
                "id": session.id,
                "session_id": session.session_id,
                "initial_message": session.initial_message,
                "updated_time": session.updated_time,
            }
            for session in sessions
        ]
    finally:
        session.close()

    return sessions_info

def get_session_by_session_id(session_id):
    session = MessageHistorySessionLocal()
    try:
        session_data = session.query(MessageHistory).filter_by(session_id=session_id).first()
        if session_data:
            session_info = {
                "id": session_data.id,
                "session_id": session_data.session_id,
                "username": session_data.username,
                "model_name": session_data.model_name,
                "initial_message": session_data.initial_message,
                "updated_time": session_data.updated_time,
                "messages": session_data.messages,
            }
        else:
            session_info = None
    finally:
        session.close()

    return session_info

def get_session_by_id(session_id: int):
    session = MessageHistorySessionLocal()
    try:
        session_data = session.query(MessageHistory).filter_by(id=session_id).first()
        if session_data:
            session_info = {
                "id": session_data.id,
                "session_id": session_data.session_id,
                "username": session_data.username,
                "model_name": session_data.model_name,
                "initial_message": session_data.initial_message,
                "updated_time": session_data.updated_time,
                "messages": session_data.messages,
            }
        else:
            session_info = None
    finally:
        session.close()

    return session_info


def archive_session_by_id(session_id: int):
    session = MessageHistorySessionLocal()
    try:
        session_data = session.query(MessageHistory).filter_by(id=session_id).first()
        session_data.archived = True
        session.commit()
    finally:
        session.close()


def update_or_create_rawdata(
    session_id, message_id, username, messages_in_train_format, updated_time
):
    session = RawDataSessionLocal()

    # Attempt to find an existing session with the same session_id and username
    existing_message_raw_data = (
        session.query(RawData)
        .filter_by(session_id=session_id, message_id=message_id, username=username)
        .first()
    )

    if existing_message_raw_data:
        # If the session exists, update it
        existing_message_raw_data.raw_data = json.dumps(messages_in_train_format)
        existing_message_raw_data.updated_time = updated_time
    else:
        # If the session does not exist, create a new one
        new_message_raw_data = RawData(
            session_id=session_id,
            message_id=message_id,
            username=username,
            raw_data=json.dumps(messages_in_train_format),
            created_time=updated_time,
            updated_time=updated_time,
        )
        session.add(new_message_raw_data)

    # Commit the changes to the database
    session.commit()
    session.close()


def get_rawdata_by_message_id(message_id: str):
    print("get_rawdata_by_message_id: ", message_id)
    session = RawDataSessionLocal()
    try:
        raw_datas = (
            session.query(RawData)
            .filter_by(message_id=message_id)
            .with_entities(RawData.session_id, RawData.message_id, RawData.raw_data)
            .all()
        )
        raw_data_info = [
            {
                "session_id": raw_data.session_id,
                "message_id": raw_data.message_id,
                "raw_data": raw_data.raw_data,
            }
            for raw_data in raw_datas
        ]
    finally:
        session.close()

    return raw_data_info


def get_rawdata_by_session_id(session_id: str):
    print("get_rawdata_by_session_id: ", session_id)
    session = RawDataSessionLocal()
    try:
        raw_datas = (
            session.query(RawData)
            .filter_by(session_id=session_id)
            .with_entities(RawData.session_id, RawData.message_id, RawData.raw_data)
            .all()
        )
        raw_data_info = [
            {
                "session_id": raw_data.session_id,
                "message_id": raw_data.message_id,
                "raw_data": raw_data.raw_data,
            }
            for raw_data in raw_datas
        ]
    finally:
        session.close()

    return raw_data_info


def update_or_create_annotation(
    session_id,
    message_id,
    username,
    tag,
    for_evaluation,
    old_message,
    suggestion,
    messages_in_train_format,
    updated_time,
):
    session = AnnotationSessionLocal()
    existing_message_annotation = (
        session.query(Annotation)
        .filter_by(session_id=session_id, message_id=message_id, username=username)
        .first()
    )
    print("old_message: ", old_message)
    print("suggestion: ", suggestion)
    print("messages_in_train_format: ")
    print(messages_in_train_format)

    if existing_message_annotation:
        print("existing_message_annotation: ", existing_message_annotation)
        # If the session exists, update it
        existing_message_annotation.tag = tag
        existing_message_annotation.for_evaluation = for_evaluation
        existing_message_annotation.old_message = old_message
        existing_message_annotation.suggestion = suggestion
        existing_message_annotation.annotations = messages_in_train_format
        existing_message_annotation.updated_time = updated_time
    else:
        # If the session does not exist, create a new one
        new_message_annotation = Annotation(
            session_id=session_id,
            message_id=message_id,
            username=username,
            tag=tag,
            for_evaluation=for_evaluation,
            old_message=old_message,
            suggestion=suggestion,
            annotations=messages_in_train_format,
            created_time=updated_time,
            updated_time=updated_time,
        )
        session.add(new_message_annotation)

    # Commit the changes to the database
    session.commit()

    # After updating or creating, count how many records the user has
    user_record_count = session.query(Annotation).filter_by(username=username).count()

    # Close the session
    session.close()

    print("Annotation updated or created successfully.")
    print("username: ", username)
    print("updated time: ", updated_time)
    print(f"User '{username}' has {user_record_count} records in the database.")


def get_all_annotations():
    session = AnnotationSessionLocal()
    try:
        annotations = (
            session.query(Annotation).order_by(Annotation.updated_time.desc()).all()
        )

        annotations_info = [
            {
                "session_id": annotation.session_id,
                "message_id": annotation.message_id,
                "username": annotation.username,
                "tag": annotation.tag,
                "for_evaluation": annotation.for_evaluation,
                "old_message": annotation.old_message,
                "suggestion": annotation.suggestion,
                "annotations": annotation.annotations,
                "created_time": annotation.created_time,  # 转换为ISO格式字符串
                "updated_time": annotation.updated_time,  # 转换为ISO格式字符串
            }
            for annotation in annotations
        ]
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        session.close()
    # print(annotations_info)
    print("number of found annotation: ", len(annotations_info))
    return annotations_info


def get_annotations_by_username_and_date_range(
    username: str, start_date: str, end_date: str
):
    session = AnnotationSessionLocal()
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
    print("username: ", username)
    print("start_date_obj: ", start_date_obj)
    print("end_date_obj: ", end_date_obj)

    try:
        annotations = (
            session.query(Annotation)
            .filter(
                Annotation.username == username,
                func.date(Annotation.updated_time) >= start_date_obj,
                func.date(Annotation.updated_time) <= end_date_obj,
            )
            .order_by(Annotation.updated_time.desc())
            .all()
        )

        annotations_info = [
            {
                "session_id": annotation.session_id,
                "message_id": annotation.message_id,
                "username": annotation.username,
                "tag": annotation.tag,
                "for_evaluation": annotation.for_evaluation,
                "old_message": annotation.old_message,
                "suggestion": annotation.suggestion,
                "annotations": annotation.annotations,
                "created_time": annotation.created_time,  # 转换为ISO格式字符串
                "updated_time": annotation.updated_time,  # 转换为ISO格式字符串
            }
            for annotation in annotations
        ]
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        session.close()
    # print(annotations_info)
    print("number of found annotation: ", len(annotations_info))

    return annotations_info


def get_annotations_for_evaluation():
    session = AnnotationSessionLocal()
    annotations = (
        session.query(Annotation)
        .filter(Annotation.for_evaluation == True)
        .order_by(Annotation.updated_time.desc())
        .all()
    )
    annotations_info = [
        {
            "session_id": annotation.session_id,
            "message_id": annotation.message_id,
            "username": annotation.username,
            "tag": annotation.tag,
            "for_evaluation": annotation.for_evaluation,
            "old_message": annotation.old_message,
            "suggestion": annotation.suggestion,
            "annotations": annotation.annotations,
            "created_time": annotation.created_time,
            "updated_time": annotation.updated_time,
        }
        for annotation in annotations
    ]
    return annotations_info


def get_annotation_counts_by_username(username: str, current_time: str):
    session = AnnotationSessionLocal()

    print(current_time)

    today = datetime.strptime(current_time, "%Y-%m-%dT%H:%M:%S.%fZ").date()

    this_week_start = today - relativedelta(weekday=MO(-1))

    try:
        count_today = (
            session.query(Annotation)
            .filter(
                Annotation.username == username,
                func.date(Annotation.updated_time) >= today,
            )
            .count()
        )

        count_this_week = (
            session.query(Annotation)
            .filter(
                Annotation.username == username,
                func.date(Annotation.updated_time) >= this_week_start,
            )
            .count()
        )

        total_count = (
            session.query(Annotation).filter(Annotation.username == username).count()
        )

        return {
            "today": count_today,
            "this_week": count_this_week,
            "total": total_count,
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        session.close()
