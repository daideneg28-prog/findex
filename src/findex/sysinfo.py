"""Process resource usage without third-party packages (for the Lab 5 benchmark).

* peak resident set size of the current process,
* total CPU time (user + system) of another process, e.g. a pool worker.

Windows goes through the Win32 API with ctypes; Linux reads /proc and getrusage.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    class _MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    _kernel32.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_MemoryCounters),
        wintypes.DWORD,
    ]

    def _filetime_s(ft: wintypes.FILETIME) -> float:
        return ((ft.dwHighDateTime << 32) | ft.dwLowDateTime) / 1e7  # 100 ns units

    def peak_rss_mib() -> float:
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = _kernel32.GetCurrentProcess()
        if not _kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return float("nan")
        return counters.PeakWorkingSetSize / 2**20

    def process_cpu_seconds(pid: int) -> float:
        handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return float("nan")
        try:
            times = [wintypes.FILETIME() for _ in range(4)]  # creation, exit, kernel, user
            if not _kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                return float("nan")
            return _filetime_s(times[2]) + _filetime_s(times[3])
        finally:
            _kernel32.CloseHandle(handle)

else:
    import resource

    def peak_rss_mib() -> float:
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return maxrss / 2**20 if sys.platform == "darwin" else maxrss / 2**10  # bytes vs KiB

    def process_cpu_seconds(pid: int) -> float:
        try:
            with open(f"/proc/{pid}/stat", encoding="ascii") as f:
                fields = f.read().rsplit(")", 1)[1].split()
        except OSError:
            return float("nan")
        ticks = int(fields[11]) + int(fields[12])  # utime + stime (fields 14 and 15)
        return ticks / os.sysconf("SC_CLK_TCK")


def cpu_description() -> str:
    """Best-effort CPU name and core count for the benchmark header."""
    name = ""
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    else:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return f"{name or 'unknown CPU'}, {os.cpu_count()} logical CPUs"
