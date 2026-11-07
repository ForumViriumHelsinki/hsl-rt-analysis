# Use Python 3.14 slim base image
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# TOPICS (whitespace-separated MQTT topics) has no default and must be set at run time.
ENV LOG=INFO
ENV OUTPUT_DIR=/app/data

# Install only required system dependencies
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    gcc \
#    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files
COPY requirements.txt /app/requirements.txt
COPY hfp_mqtt_data_dumper.py /app/hfp_mqtt_data_dumper.py

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the MQTT dumper script. Shell form so the variables expand; exec keeps
# Python as PID 1 so it receives SIGTERM. $TOPICS is intentionally unquoted so
# it splits on whitespace into one argument per topic.
CMD exec python hfp_mqtt_data_dumper.py --output-dir "$OUTPUT_DIR" --log "$LOG" --topics ${TOPICS:?TOPICS must be set to a whitespace-separated list of MQTT topics}
