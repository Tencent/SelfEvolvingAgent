FROM python:3.9

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt
COPY . /app

RUN mkdir /app/Databases
RUN mkdir /app/Logs
RUN mkdir /app/static
RUN mkdir /app/static/avatar
RUN mkdir /app/static/uploads
RUN mkdir /app/Database_global
ENV PYTHONPATH "${PYTHONPATH}:/app"
# CMD ["pip", "freeze"]
CMD ["python", "main.py"]
# CMD ["python", "ftq_search_test_main.py"]