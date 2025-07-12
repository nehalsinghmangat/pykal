import sys
from mcap.reader import make_reader
import numpy as np
import pandas as pd
from collections import defaultdict
from io import BufferedReader
from typing import Dict


def extract_float64multiarray(msg_bytes: bytes) -> list[float]:
    """Extract Float64MultiArray from raw msg bytes (assuming no layout)."""
    from std_msgs.msg import Float64MultiArray
    import rclpy.serialization

    return rclpy.serialization.deserialize_message(msg_bytes, Float64MultiArray).data


def read_mcap_to_dataframes(mcap_path: str) -> Dict[str, pd.DataFrame]:
    topic_data = defaultdict(list)
    topic_times = defaultdict(list)

    with open(mcap_path, "rb") as f:
        reader = make_reader(BufferedReader(f))
        for schema, channel, message in reader.iter_decoded_messages():
            if channel.topic in ("/sim/state", "/sim/meas"):
                topic_data[channel.topic].append(
                    np.frombuffer(message.data, dtype=np.float64)
                )
                topic_times[channel.topic].append(message.log_time / 1e9)  # ns → s

    dfs = {}
    for topic, data_list in topic_data.items():
        data_array = np.array(data_list)
        times = topic_times[topic]
        df = pd.DataFrame(data_array, index=pd.Index(times, name="time"))
        df.columns = [f"{topic[5:]}_{i}" for i in range(data_array.shape[1])]
        dfs[topic] = df

    return dfs


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_mcap_data.py <your_file.mcap>")
        sys.exit(1)

    path = sys.argv[1]
    dataframes = read_mcap_to_dataframes(path)

    for topic, df in dataframes.items():
        print(f"\nTopic: {topic}")
        print(df.head())
