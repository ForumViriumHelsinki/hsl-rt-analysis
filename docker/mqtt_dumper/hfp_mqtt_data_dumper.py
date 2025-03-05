import argparse
import datetime
import gzip
import json
import logging
import os
import shutil
import threading
from datetime import timedelta
from pathlib import Path
from typing import Dict

import paho.mqtt.client as mqtt


"""
example:
--topics "/hfp/v2/journey/ongoing/+/tram/+/+/2015/#" "/hfp/v2/journey/ongoing/+/tram/+/+/1007/#" "/hfp/v2/journey/ongoing/+/tram/+/+/1009B/#" "/hfp/v2/journey/ongoing/+/tram/+/+/1001T/#" "/hfp/v2/journey/ongoing/+/tram/+/+/1008/#"  "/hfp/v2/journey/ongoing/+/tram/+/+/1009/#"
"""

# MQTT connection status codes
MQTT_RC_CODES = {
    0: "Connection successful",
    1: "Connection refused - incorrect protocol version",
    2: "Connection refused - invalid client identifier",
    3: "Connection refused - server unavailable",
    4: "Connection refused - bad username or password",
    5: "Connection refused - not authorised",
}

# Default settings
DEFAULT_RECONNECT_DELAY = 30  # seconds
DEFAULT_QOS = 1  # At least once delivery
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
DEFAULT_ROTATION_INTERVAL = 24  # hours


class FileHandler:
    """Class to handle file operations with rotation and compression."""

    def __init__(self, base_dir: Path, max_size: int, rotation_interval: int):
        self.base_dir = base_dir
        self.max_size = max_size
        self.rotation_interval = datetime.timedelta(hours=rotation_interval)

    def get_filepath(self, route: str, date: datetime.datetime) -> Path:
        """Get the appropriate file path for the given route and date."""
        date_str = date.strftime("%Y-%m-%d")

        # Create directory structure: output_dir/date/route-YYYY-MM-DDTHH.txt
        file_dir = self.base_dir / date_str
        file_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{route}-{date.strftime('%Y-%m-%dT%H')}.txt"
        return file_dir / filename

    def rotate_file(self, filepath: Path):
        """Compress and rotate the file."""
        if not filepath.exists():
            return

        # Compress the file
        compressed_path = filepath.with_suffix(".txt.gz")
        with open(filepath, "rb") as f_in:
            with gzip.open(compressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the original file
        filepath.unlink()
        logging.info(f"Rotated and compressed file: {compressed_path}")


class RotationManager:
    """Manages periodic file rotation and compression."""

    def __init__(self, base_dir: Path, check_interval: timedelta, min_age: timedelta, file_handler: FileHandler):
        """
        Args:
            base_dir: Directory to monitor for files
            check_interval: How often to check for files to rotate
            min_age: Minimum age of file before rotation
            file_handler: FileHandler instance for rotation operations
        """
        self.base_dir = base_dir
        self.check_interval = check_interval
        self.min_age = min_age
        self.file_handler = file_handler
        self._timer = None
        self._running = False

    def start(self):
        """Start the rotation manager with immediate first check in separate thread."""
        self._running = True
        # Start immediate first check in separate thread
        initial_check = threading.Thread(target=self._initial_check)
        initial_check.daemon = True
        initial_check.start()
        logging.info(f"RotationManager started. Checking every {self.check_interval}")

    def _initial_check(self):
        """Perform initial check and schedule periodic checks."""
        self._check_files()  # Run first check immediately
        self._schedule_next_check()  # Schedule next periodic check

    def stop(self):
        """Stop the rotation manager."""
        self._running = False
        if self._timer:
            self._timer.cancel()
        logging.info("RotationManager stopped")

    def _schedule_next_check(self):
        """Schedule the next file check."""
        if self._running:
            self._timer = threading.Timer(self.check_interval.total_seconds(), self._check_files)
            self._timer.daemon = True  # Allow program to exit if this is running
            self._timer.start()

    def _check_files(self):
        """Check and rotate old files."""
        try:
            now = datetime.datetime.now()
            # Find all .txt files recursively
            for txt_file in self.base_dir.rglob("*.txt"):
                try:
                    # Get file's last modification time
                    mtime = datetime.datetime.fromtimestamp(txt_file.stat().st_mtime)
                    age = now - mtime

                    # Rotate if file is old enough
                    if age >= self.min_age:
                        logging.info(f"Rotating old file: {txt_file} (age: {age})")
                        self.file_handler.rotate_file(txt_file)
                except Exception as e:
                    logging.error(f"Error processing file {txt_file}: {e}")

        except Exception as e:
            logging.error(f"Error during file rotation check: {e}")
        finally:
            self._schedule_next_check()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments and configure logging.

    Returns:
        Arguments: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="MQTT Subscriber to log HSL vehicle data", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # MQTT connection arguments
    mqtt_group = parser.add_argument_group("MQTT Connection Settings")
    mqtt_group.add_argument("--broker-host", default="mqtt.hsl.fi", help="MQTT broker hostname")
    mqtt_group.add_argument("--broker-port", type=int, default=8883, help="MQTT broker port")
    mqtt_group.add_argument("--keepalive", type=int, default=60, help="MQTT keepalive interval in seconds")
    mqtt_group.add_argument("--username", help="MQTT username if authentication is required")
    mqtt_group.add_argument(
        "--password",
        help="MQTT password if authentication is required. Can also be set via MQTT_PASSWORD environment variable",
    )
    mqtt_group.add_argument("--client-id", help="MQTT client ID. If not provided, a random one will be generated")
    mqtt_group.add_argument("--no-tls", action="store_true", help="Disable TLS encryption")
    mqtt_group.add_argument(
        "--qos", type=int, choices=[0, 1, 2], default=DEFAULT_QOS, help="MQTT Quality of Service level"
    )
    mqtt_group.add_argument(
        "--reconnect-delay",
        type=int,
        default=DEFAULT_RECONNECT_DELAY,
        help="Delay in seconds between reconnection attempts",
    )

    # Data collection arguments
    data_group = parser.add_argument_group("Data Collection Settings")
    data_group.add_argument("--topics", nargs="+", required=True, help="List of MQTT topics to subscribe to")
    data_group.add_argument("--output-dir", type=Path, required=True, help="Directory to store output files")
    data_group.add_argument(
        "--max-file-size",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE,
        help="Maximum file size in bytes before rotation",
    )
    data_group.add_argument(
        "--rotation-interval",
        type=int,
        default=DEFAULT_ROTATION_INTERVAL,
        help="Time in hours before file rotation",
    )

    # Logging arguments
    log_group = parser.add_argument_group("Logging Settings")
    log_group.add_argument(
        "--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set logging level"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=args.log, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check for password in environment variable
    if not args.password and args.username:
        args.password = os.environ.get("MQTT_PASSWORD")
        if not args.password:
            logging.warning(
                "Username provided but password not found in arguments or MQTT_PASSWORD environment variable"
            )

    return args


def on_connect(client: mqtt.Client, userdata: Dict, flags: Dict, rc: int):
    """Callback for when the client connects to the broker."""

    if rc == 0:
        logging.info("Connected to MQTT broker successfully")
        # Subscribe to all topics
        for topic in userdata["topics"]:
            client.subscribe(topic, qos=userdata["qos"])
            logging.info(f"Subscribed to topic: {topic} with QoS {userdata['qos']}")
    else:
        logging.error(f"Connection failed: {MQTT_RC_CODES.get(rc, f'Unknown error ({rc})')}")


def on_disconnect(client: mqtt.Client, userdata: Dict, rc: int):
    """Callback for when the client disconnects from the broker."""
    if rc != 0:
        logging.warning("Unexpected disconnection, will automatically reconnect...")
    else:
        logging.info("Disconnected from broker")


def on_message(client: mqtt.Client, userdata: Dict, msg: mqtt.MQTTMessage):
    """Callback for when a message is received from the broker."""
    try:
        # Decode the message payload
        payload = json.loads(msg.payload.decode("utf-8"))
        logging.debug(f"Received message on topic {msg.topic}")

        # Extract route and get current time
        topic_parts = msg.topic.split("/")
        route = topic_parts[9]
        current_time = datetime.datetime.now(datetime.UTC)

        # Get file path and handle rotation
        file_handler = userdata["file_handler"]
        filepath = file_handler.get_filepath(route, current_time)

        # Prepare the message
        message = f"{msg.topic} {json.dumps(payload)}\n"

        # Write to file
        with open(filepath, "a") as f:
            f.write(message)

        logging.debug(f"Data written to {filepath}")

    except Exception as e:
        logging.error(f"Error processing message: {e}")


def setup_mqtt_client(args: argparse.Namespace) -> mqtt.Client:
    """Setup and configure MQTT client with the given arguments."""
    # Create client instance
    client_kwargs = {}
    if args.client_id:
        client_kwargs["client_id"] = args.client_id

    file_handler = FileHandler(args.output_dir, args.max_file_size, args.rotation_interval)

    client = mqtt.Client(
        userdata={
            "topics": args.topics,
            "output_dir": args.output_dir,
            "qos": args.qos,
            "file_handler": file_handler,
        },
        **client_kwargs,
    )

    # Set callbacks
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # Configure automatic reconnect
    client.reconnect_delay_set(min_delay=1, max_delay=args.reconnect_delay)

    # Configure authentication if provided
    if args.username:
        client.username_pw_set(args.username, args.password)

    # Configure TLS unless disabled
    if not args.no_tls:
        client.tls_set()

    return client


def main():
    # Parse arguments and configure logging
    args = parse_args()

    # Setup MQTT client
    client = setup_mqtt_client(args)

    # Create and start rotation manager
    file_handler = client._userdata["file_handler"]
    rotation_manager = RotationManager(
        base_dir=args.output_dir,
        check_interval=timedelta(hours=1),  # Check every hour
        min_age=timedelta(hours=3),  # Rotate files older than 3 hours
        file_handler=file_handler,
    )

    try:
        # Start rotation manager
        rotation_manager.start()

        # Connect to the MQTT broker
        logging.info(f"Connecting to broker {args.broker_host}:{args.broker_port}...")
        client.connect(args.broker_host, args.broker_port, args.keepalive)

        logging.info("Starting MQTT client loop...")
        client.loop_forever()

    except KeyboardInterrupt:
        logging.info("Shutting down...")
        rotation_manager.stop()
        client.disconnect()
    except ConnectionRefusedError:
        logging.error(f"Connection refused to {args.broker_host}:{args.broker_port}")
        rotation_manager.stop()
        raise
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
        rotation_manager.stop()
        raise


if __name__ == "__main__":
    main()
