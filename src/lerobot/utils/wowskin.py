# Active WowSkin helpers used by lerobot-record.
#
# Why: the repository keeps the old prototype below for reference, but the
# recording path needs importable helpers now so force data can be recorded as
# observation.force and visualized in the dataset viewer.

from __future__ import annotations

import importlib
import struct
import threading
import time
from typing import Any

import numpy as np


def _serial_module():
	return importlib.import_module("serial")


class AnySkinBase:
	"""Direct serial reader for a WowSkin/AnySkin sensor."""

	def __init__(
		self,
		num_mags: int = 1,
		port: str | None = None,
		device_id: int = -1,
		temp_filtered: bool = True,
		burst_mode: bool = True,
		baudrate: int = 115200,
	) -> None:
		serial = _serial_module()

		self.num_mags = num_mags
		self.port_name = port
		self.baud_rate = baudrate
		self.burst_mode = burst_mode
		self.device_id = device_id

		self._msg_floats = 4 * num_mags
		self._msg_length = 4 * self._msg_floats + 2

		self._temp_mask = np.ones((self._msg_floats,), dtype=bool)
		if temp_filtered:
			self._temp_mask[::4] = False

		self._serial = serial.Serial(port=port, baudrate=baudrate)
		self._initialize()

	@property
	def in_waiting(self) -> int:
		return self._serial.in_waiting

	def flush(self) -> None:
		self._serial.flush()

	def reset_input_buffer(self) -> None:
		self._serial.reset_input_buffer()

	def read(self, size: int) -> bytes:
		return self._serial.read(size)

	def read_until(self, expected: bytes = b"\n") -> bytes:
		return self._serial.read_until(expected)

	def readline(self) -> bytes:
		return self._serial.readline()

	def close(self) -> None:
		self._serial.close()

	def _initialize(self) -> None:
		self.flush()
		try:
			self.get_sample()
		except Exception:  # noqa: BLE001
			pass

	def get_data(self, num_samples: int) -> list[Any]:
		data = []
		for _ in range(num_samples):
			t, sample = self.get_sample()
			data.append(np.concatenate(([t], sample)))
		return data

	def get_sample(self) -> tuple[float, Any]:
		if self.in_waiting > 4000:
			self.reset_input_buffer()
			while True:
				if self.in_waiting > self._msg_length:
					if self.read(self._msg_length)[-2:] == b"\r\n":
						break
					self.reset_input_buffer()

		while True:
			if self.in_waiting > self._msg_length:
				collect_start = time.time()
				if self.burst_mode:
					zero_bytes = self.read(self._msg_length)
					if zero_bytes[-2:] != b"\r\n":
						zero_bytes = self.read_until(b"\r\n")
						continue
					decoded_zero_bytes = struct.unpack("@{}fcc".format(self._msg_floats), zero_bytes)[
						: self._msg_floats
					]
				else:
					zero_bytes = self.readline()
					decoded_zero_bytes = zero_bytes.decode("utf-8").strip().split()
					decoded_zero_bytes = [float(x) for x in decoded_zero_bytes]
				return collect_start, np.array(decoded_zero_bytes)[self._temp_mask]

			time.sleep(0.001)


class AnySkinDummy(AnySkinBase):
	"""Fallback sensor that produces random force channels."""

	def __init__(
		self,
		num_mags: int = 1,
		port: str | None = None,
		baudrate: int = 115200,
		burst_mode: bool = True,
		device_id: int = -1,
		temp_filtered: bool = False,
	):
		self.num_mags = num_mags
		self.port_name = port
		self.baud_rate = baudrate
		self.burst_mode = burst_mode
		self.device_id = device_id

		self._msg_floats = 4 * num_mags
		self._msg_length = 4 * self._msg_floats + 2

		self._temp_mask = np.ones((self._msg_floats,), dtype=bool)
		if temp_filtered:
			self._temp_mask[::4] = False

	def _initialize(self) -> None:
		pass

	def get_sample(self) -> tuple[float, Any]:
		collect_start = time.time()
		data = np.random.uniform(-1.0, 1.0, size=(np.sum(self._temp_mask),))
		return collect_start, data


class AnySkinProcess:
	"""Background sampler that keeps the most recent WowSkin reading available."""

	def __init__(
		self,
		num_mags: int = 1,
		port: str | None = None,
		device_id: int = -1,
		temp_filtered: bool = True,
		burst_mode: bool = True,
		baudrate: int = 115200,
		use_dummy: bool = False,
	) -> None:
		self._sensor = (
			AnySkinDummy(
				num_mags=num_mags,
				port=port,
				device_id=device_id,
				temp_filtered=temp_filtered,
				burst_mode=burst_mode,
				baudrate=baudrate,
			)
			if use_dummy or not port
			else AnySkinBase(
				num_mags=num_mags,
				port=port,
				device_id=device_id,
				temp_filtered=temp_filtered,
				burst_mode=burst_mode,
				baudrate=baudrate,
			)
		)
		self._lock = threading.Lock()
		self._stop_event = threading.Event()
		self._thread = threading.Thread(target=self._stream_loop, daemon=True)
		self._latest_sample: tuple[float, Any] | None = None
		self._started = False

	def start(self) -> None:
		if self._started:
			return
		self._started = True
		self._stop_event.clear()
		self._thread.start()

	def _stream_loop(self) -> None:
		while not self._stop_event.is_set():
			try:
				sample = self._sensor.get_sample()
			except Exception:  # noqa: BLE001
				time.sleep(0.05)
				continue

			with self._lock:
				self._latest_sample = sample

	def get_sample(self) -> tuple[float, Any]:
		if not self._started:
			self.start()

		while True:
			with self._lock:
				if self._latest_sample is not None:
					return self._latest_sample
			time.sleep(0.005)

	def pause_streaming(self) -> None:
		self._stop_event.set()

	def join(self) -> None:
		if self._thread.is_alive():
			self._thread.join()

	def close(self) -> None:
		self.pause_streaming()
		self.join()
		self._sensor.close()


def wowskin_force_dim(num_mags: int, temp_filtered: bool = True) -> int:
	return num_mags * (3 if temp_filtered else 4)


def wowskin_force_feature_names(
	num_mags: int,
	temp_filtered: bool = True,
	feature_prefix: str = "force",
) -> list[str]:
	return [f"{feature_prefix}_{idx}" for idx in range(wowskin_force_dim(num_mags, temp_filtered))]


def wowskin_force_feature_spec(
	num_mags: int,
	temp_filtered: bool = True,
	feature_name: str = "observation.force",
	feature_prefix: str = "force",
) -> dict[str, dict[str, Any]]:
	names = wowskin_force_feature_names(num_mags, temp_filtered, feature_prefix=feature_prefix)
	return {
		feature_name: {
			"dtype": "float32",
			"shape": (len(names),),
			"names": names,
		}
	}


def wowskin_sample_to_values(
	sample: Any,
	feature_prefix: str = "force",
) -> dict[str, float]:
	flat_sample = np.asarray(sample).reshape(-1)
	return {f"{feature_prefix}_{idx}": float(value) for idx, value in enumerate(flat_sample)}


# """WowSkin sensor helpers.

# This is a small, self-contained port of the WowSkin/AnySkin reader so the
# recording pipeline can sample tactile channels without depending on the
# standalone UI package at runtime.
# """

# from __future__ import annotations

# import importlib
# import struct
# import threading
# import time
# from typing import Any


# def _np_module():
#     return importlib.import_module("numpy")


# def _serial_module():
#     return importlib.import_module("serial")


# class AnySkinBase:
#     """Direct serial reader for a WowSkin/AnySkin sensor."""

#     def __init__(
#         self,
#         num_mags: int = 1,
#         port: str | None = None,
#         device_id: int = -1,
#         temp_filtered: bool = True,
#         burst_mode: bool = True,
#         baudrate: int = 115200,
#     ) -> None:
#         np = _np_module()
#         serial = _serial_module()

#         self.num_mags = num_mags
#         self.port_name = port
#         self.baud_rate = baudrate
#         self.burst_mode = burst_mode
#         self.device_id = device_id

#         self._msg_floats = 4 * num_mags
#         self._msg_length = 4 * self._msg_floats + 2

#         self._temp_mask = np.ones((self._msg_floats,), dtype=bool)
#         if temp_filtered:
#             self._temp_mask[::4] = False

#         self._serial = serial.Serial(port=port, baudrate=baudrate)
#         self._initialize()

#     @property
#     def in_waiting(self) -> int:
#         return self._serial.in_waiting

#     def flush(self) -> None:
#         self._serial.flush()

#     def reset_input_buffer(self) -> None:
#         self._serial.reset_input_buffer()

#     def read(self, size: int) -> bytes:
#         return self._serial.read(size)

#     def read_until(self, expected: bytes = b"\n") -> bytes:
#         return self._serial.read_until(expected)

#     def readline(self) -> bytes:
#         return self._serial.readline()

#     def close(self) -> None:
#         self._serial.close()

#     def _initialize(self) -> None:
#         """Open the serial link and confirm the sensor starts streaming."""
#         self.flush()
#         print("Initializing WowSkin sensor...")
#         try:
#             self.get_sample()
#             print("WowSkin initialization successful")
#         except Exception as exc:  # noqa: BLE001
#             print(f"WowSkin initialization failed with error: {exc}")

#     def get_data(self, num_samples: int) -> list[Any]:
#         """Collect ``num_samples`` consecutive samples from the sensor."""
#         np = _np_module()
#         data = []
#         for _ in range(num_samples):
#             t, sample = self.get_sample()
#             data.append(np.concatenate(([t], sample)))

#         return data

#     def get_sample(self) -> tuple[float, Any]:
#         """Collect one sample from the serial stream."""
#         np = _np_module()
#         if self.in_waiting > 4000:
#             self.reset_input_buffer()
#             while True:
#                 if self.in_waiting > self._msg_length:
#                     if self.read(self._msg_length)[-2:] == b"\r\n":
#                         break
#                     self.reset_input_buffer()

#         while True:
#             if self.in_waiting > self._msg_length:
#                 collect_start = time.time()
#                 if self.burst_mode:
#                     zero_bytes = self.read(self._msg_length)
#                     if zero_bytes[-2:] != b"\r\n":
#                         zero_bytes = self.read_until(b"\r\n")
#                         continue
#                     decoded_zero_bytes = struct.unpack("@{}fcc".format(self._msg_floats), zero_bytes)[
#                         : self._msg_floats
#                     ]
#                 else:
#                     zero_bytes = self.readline()
#                     decoded_zero_bytes = zero_bytes.decode("utf-8")
#                     decoded_zero_bytes = decoded_zero_bytes.strip()
#                     decoded_zero_bytes = [float(x) for x in decoded_zero_bytes.split()]
#                 return collect_start, np.array(decoded_zero_bytes)[self._temp_mask]

#             time.sleep(0.001)


# class AnySkinDummy(AnySkinBase):
#     """Fallback sensor that produces random force channels."""

#     def __init__(
#         self,
#         num_mags: int = 1,
#         port: str | None = None,
#         baudrate: int = 115200,
#         burst_mode: bool = True,
#         device_id: int = -1,
#         temp_filtered: bool = False,
#     ):
#         self.num_mags = num_mags
#         self.port_name = port
#         self.baud_rate = baudrate
#         self.burst_mode = burst_mode
#         self.device_id = device_id

#         self._msg_floats = 4 * num_mags
#         self._msg_length = 4 * self._msg_floats + 2

#         self._temp_mask = np.ones((self._msg_floats,), dtype=bool)
#         if temp_filtered:
#             self._temp_mask[::4] = False

#     def _initialize(self) -> None:
#         pass

#     def get_sample(self) -> tuple[float, Any]:
#         np = _np_module()
#         collect_start = time.time()
#         data = np.random.uniform(-1.0, 1.0, size=(np.sum(self._temp_mask),))
#         return collect_start, data


# class AnySkinProcess:
#     """Background sampler that keeps the most recent WowSkin reading available.

#     lerobot-record uses this wrapper so force data can be read without blocking
#     the main control loop.
#     """

#     def __init__(
#         self,
#         num_mags: int = 1,
#         port: str | None = None,
#         device_id: int = -1,
#         temp_filtered: bool = True,
#         burst_mode: bool = True,
#         baudrate: int = 115200,
#         use_dummy: bool = False,
#     ) -> None:
#         self._sensor = (
#             AnySkinDummy(
#                 num_mags=num_mags,
#                 port=port,
#                 device_id=device_id,
#                 temp_filtered=temp_filtered,
#                 burst_mode=burst_mode,
#                 baudrate=baudrate,
#             )
#             if use_dummy or not port
#             else AnySkinBase(
#                 num_mags=num_mags,
#                 port=port,
#                 device_id=device_id,
#                 temp_filtered=temp_filtered,
#                 burst_mode=burst_mode,
#                 baudrate=baudrate,
#             )
#         )
#         self._lock = threading.Lock()
#         self._stop_event = threading.Event()
#         self._thread = threading.Thread(target=self._stream_loop, daemon=True)
#         self._latest_sample: tuple[float, Any] | None = None
#         self._started = False

#     def start(self) -> None:
#         if self._started:
#             return
#         self._started = True
#         self._stop_event.clear()
#         self._thread.start()

#     def _stream_loop(self) -> None:
#         while not self._stop_event.is_set():
#             try:
#                 sample = self._sensor.get_sample()
#             except Exception as exc:  # noqa: BLE001
#                 print(f"WowSkin stream read failed: {exc}")
#                 time.sleep(0.05)
#                 continue

#             with self._lock:
#                 self._latest_sample = sample

#     def get_sample(self) -> tuple[float, Any]:
#         if not self._started:
#             self.start()

#         while True:
#             with self._lock:
#                 if self._latest_sample is not None:
#                     return self._latest_sample
#             time.sleep(0.005)

#     def pause_streaming(self) -> None:
#         self._stop_event.set()

#     def join(self) -> None:
#         if self._thread.is_alive():
#             self._thread.join()

#     def close(self) -> None:
#         self.pause_streaming()
#         self.join()
#         self._sensor.close()


# def wowskin_force_dim(num_mags: int, temp_filtered: bool = True) -> int:
#     """Return the number of force channels exposed by the sensor."""

#     # The raw reader emits 4 values per magnetometer and optionally drops the
#     # temperature channel; the recording pipeline stores the remaining channels.
#     return num_mags * (3 if temp_filtered else 4)


# def wowskin_force_feature_names(
#     num_mags: int,
#     temp_filtered: bool = True,
#     feature_prefix: str = "wowskin_force",
# ) -> list[str]:
#     """Generate stable column names for the recorded WowSkin channels."""

#     return [f"{feature_prefix}_{idx}" for idx in range(wowskin_force_dim(num_mags, temp_filtered))]


# def wowskin_force_feature_spec(
#     num_mags: int,
#     temp_filtered: bool = True,
#     feature_name: str = "observation.wowskin_force",
#     feature_prefix: str = "wowskin_force",
# ) -> dict[str, dict[str, Any]]:
#     """Build the LeRobot dataset feature description for the force vector."""

#     names = wowskin_force_feature_names(num_mags, temp_filtered, feature_prefix=feature_prefix)
#     return {
#         feature_name: {
#             "dtype": "float32",
#             "shape": (len(names),),
#             "names": names,
#         }
#     }


# def wowskin_sample_to_values(
#     sample: Any,
#     feature_prefix: str = "wowskin_force",
# ) -> dict[str, float]:
#     """Convert one WowSkin sample into a flat key/value mapping."""

#     flat_sample = np.asarray(sample).reshape(-1)
#     return {f"{feature_prefix}_{idx}": float(value) for idx, value in enumerate(flat_sample)}
