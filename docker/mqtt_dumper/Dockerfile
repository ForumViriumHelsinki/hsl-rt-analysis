# Use Python 3.13 slim base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install only required system dependencies
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    gcc \
#    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files
COPY requirements.txt /app/requirements.txt
COPY hfp_mqtt_data_dumper.py /app/hfp_mqtt_data_dumper.py

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the MQTT dumper script
CMD ["python", "hfp_mqtt_data_dumper.py"]
