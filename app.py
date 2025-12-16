import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import welch, butter, filtfilt, find_peaks
import folium
import streamlit.components.v1 as components

def haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance in meters between two WGS84 points."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def moving_avg(arr: np.ndarray, w: int = 5) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(w, center=True, min_periods=1).mean().to_numpy()


def estimate_fs(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(1.0 / np.median(dt))


def choose_best_component_by_snr(acc: pd.DataFrame, fs: float) -> pd.DataFrame:
    """
    Compute a simple SNR-like score for each component:
      - signal: max PSD in 0.5–4 Hz
      - noise: median PSD in 4–10 Hz
      - score: 10*log10(signal/noise)
    Returns a dataframe sorted by score descending.
    """
    components = ["X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"]
    rows = []
    for c in components:
        f, P = welch(acc[c].to_numpy(), fs=fs, nperseg=min(4096, len(acc)))
        band = (f >= 0.5) & (f <= 4.0)
        noise_band = (f >= 4.0) & (f <= 10.0)

        if not np.any(band) or not np.any(noise_band):
            rows.append((c, np.nan, -np.inf))
            continue

        peak_idx = np.argmax(P[band])
        peak_f = float(f[band][peak_idx])
        peak_P = float(P[band][peak_idx])
        noise = float(np.median(P[noise_band]))
        snr_db = float(10 * np.log10(peak_P / noise)) if noise > 0 else float("inf")
        rows.append((c, peak_f, snr_db))

    df = pd.DataFrame(rows, columns=["component", "peak_hz_0.5_4", "snr_db_0.5_4_vs_4_10"])
    return df.sort_values("snr_db_0.5_4_vs_4_10", ascending=False).reset_index(drop=True)


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="bandpass")
    return filtfilt(b, a, x)


def count_steps_from_filtered(sig_f: np.ndarray, fs: float, min_step_interval_s: float, prom_factor: float) -> int:
    """
    Count steps by detecting both positive and negative peaks from a band-passed signal.
    This works well when gait-induced oscillation is roughly sinusoidal.
    """
    std = float(np.std(sig_f))
    if std == 0 or not np.isfinite(std):
        return 0

    min_distance = max(1, int(min_step_interval_s * fs))
    prominence = prom_factor * std

    peaks_pos, _ = find_peaks(sig_f, distance=min_distance, prominence=prominence)
    peaks_neg, _ = find_peaks(-sig_f, distance=min_distance, prominence=prominence)

    step_peaks = np.sort(np.unique(np.concatenate([peaks_pos, peaks_neg])))
    return int(len(step_peaks))


def steps_from_fft(acc_sig: np.ndarray, fs: float, duration_s: float, fmin: float = 0.7, fmax: float = 3.5):
    """
    Estimate steps using dominant PSD peak in [fmin,fmax].
    If peak < 1.5 Hz, interpret it as stride frequency and multiply by 2 for step frequency.
    """
    f, P = welch(acc_sig, fs=fs, nperseg=min(4096, len(acc_sig)))
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return 0, np.nan, np.nan, f, P

    peak_f = float(f[band][np.argmax(P[band])])

    step_freq_hz = 2 * peak_f if peak_f < 1.5 else peak_f
    steps = int(round(step_freq_hz * duration_s))

    return steps, peak_f, step_freq_hz, f, P


def gps_distance_and_speed(loc: pd.DataFrame) -> tuple[float, float]:
    """
    Return (distance_m, avg_speed_mps).
    Uses a light moving-average smoothing for lat/lon to reduce jitter.
    """
    t = loc["Time (s)"].to_numpy()
    lat = loc["Latitude (°)"].to_numpy()
    lon = loc["Longitude (°)"].to_numpy()

    lat_s = moving_avg(lat, w=5)
    lon_s = moving_avg(lon, w=5)
    mask = ~(np.isnan(lat_s) | np.isnan(lon_s) | ~np.isfinite(lat_s) | ~np.isfinite(lon_s))
    lat_s, lon_s, t_s = lat_s[mask], lon_s[mask], t[mask]

    if len(lat_s) < 2:
        return 0.0, float("nan")

    dist_m = 0.0
    for i in range(len(lat_s) - 1):
        dist_m += haversine(lat_s[i], lon_s[i], lat_s[i + 1], lon_s[i + 1])

    duration_s = float(t_s[-1] - t_s[0])
    avg_speed = float(dist_m / duration_s) if duration_s > 0 else float("nan")
    return float(dist_m), avg_speed


def folium_map_html(loc: pd.DataFrame) -> str:
    lat = loc["Latitude (°)"].to_numpy()
    lon = loc["Longitude (°)"].to_numpy()

    lat_s = moving_avg(lat, w=5)
    lon_s = moving_avg(lon, w=5)
    mask = ~(np.isnan(lat_s) | np.isnan(lon_s) | ~np.isfinite(lat_s) | ~np.isfinite(lon_s))
    lat_s, lon_s = lat_s[mask], lon_s[mask]

    if len(lat_s) < 2:
        m = folium.Map(location=[60.0, 25.0], zoom_start=12)
        return m.get_root().render()

    start_lat, start_lon = float(lat_s[0]), float(lon_s[0])
    end_lat, end_lon = float(lat_s[-1]), float(lon_s[-1])

    m = folium.Map(location=[start_lat, start_lon], zoom_start=16)
    route = list(zip(lat_s, lon_s))
    folium.PolyLine(route, weight=4).add_to(m)
    folium.Marker([start_lat, start_lon], popup="Start").add_to(m)
    folium.Marker([end_lat, end_lon], popup="Loppu").add_to(m)
    return m.get_root().render()


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

st.set_page_config(page_title="Liikunta-analyysi", layout="wide")

st.title("Päivän liikunta – kiihtyvyys- ja GPS-analyysi (Streamlit)")

st.markdown(
    """
Tämä sovellus:
- valitsee kiihtyvyyden komponentin, jossa kävely näkyy parhaiten (SNR-heuristiikka PSD:stä),
- laskee askelmäärän suodatetusta kiihtyvyysdatasta,
- laskee askelmäärän Fourier/PSD-dominantin taajuuden perusteella,
- laskee keskinopeuden ja matkan GPS-datasta,
- laskee askelpituuden.
"""
)

with st.sidebar:
    st.header("Data")
    use_uploader = st.checkbox("Lataa CSV:t käyttöliittymästä (vaihtoehto repo-/data-kansiolle)", value=False)

    if use_uploader:
        acc_file = st.file_uploader("Linear Accelerometer CSV", type=["csv"])
        loc_file = st.file_uploader("Location CSV", type=["csv"])
    else:
        acc_file = None
        loc_file = None
        st.caption("Oletuspolut: `data/Linear Accelerometer.csv` ja `data/Location.csv`")

    st.header("Askelentunnistus")
    low = st.slider("Band-pass low [Hz]", 0.2, 2.0, 0.5, 0.1)
    high = st.slider("Band-pass high [Hz]", 2.0, 8.0, 3.5, 0.1)
    min_step_interval_s = st.slider("Min askelväli [s]", 0.15, 0.8, 0.25, 0.05)
    prom_factor = st.slider("Prominenssi (kerroin * std)", 0.05, 1.0, 0.30, 0.05)

    st.header("Fourier/PSD")
    fmin = st.slider("PSD band min [Hz]", 0.2, 2.0, 0.7, 0.1)
    fmax = st.slider("PSD band max [Hz]", 2.0, 10.0, 3.5, 0.1)


try:
    if use_uploader:
        if acc_file is None or loc_file is None:
            st.warning("Lataa molemmat CSV-tiedostot sivupalkista.")
            st.stop()
        acc = pd.read_csv(acc_file)
        loc = pd.read_csv(loc_file)
    else:
        acc = load_csv("data/Linear Accelerometer.csv")
        loc = load_csv("data/Location.csv")
except Exception as e:
    st.error(f"Datan lataus epäonnistui: {e}")
    st.stop()


required_acc = {"Time (s)", "X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"}
required_loc = {"Time (s)", "Latitude (°)", "Longitude (°)"}

if not required_acc.issubset(set(acc.columns)):
    st.error(f"Linear Accelerometer CSV: puuttuvia sarakkeita. Tarvitaan: {sorted(required_acc)}")
    st.stop()

if not required_loc.issubset(set(loc.columns)):
    st.error(f"Location CSV: puuttuvia sarakkeita. Tarvitaan: {sorted(required_loc)}")
    st.stop()


t_acc = acc["Time (s)"].to_numpy()
fs = estimate_fs(t_acc)
duration_s = float(t_acc[-1] - t_acc[0]) if len(t_acc) > 1 else float("nan")

snr_df = choose_best_component_by_snr(acc, fs)
auto_component = snr_df.iloc[0]["component"]

colA, colB = st.columns([2, 1])
with colB:
    component = st.selectbox(
        "Analyysikomponentti",
        options=["X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"],
        index=["X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"].index(auto_component),
        help="Oletuksena valitaan komponentti, jolla on korkein SNR 0.5–4 Hz -kaistalla.",
    )

sig = acc[component].to_numpy()
sig_f = bandpass_filter(sig, fs, low=low, high=high)

steps_filtered = count_steps_from_filtered(
    sig_f=sig_f,
    fs=fs,
    min_step_interval_s=min_step_interval_s,
    prom_factor=prom_factor,
)

steps_fft, peak_f, step_freq_hz, f_psd, P_psd = steps_from_fft(
    acc_sig=sig,
    fs=fs,
    duration_s=duration_s,
    fmin=fmin,
    fmax=fmax,
)

dist_m, avg_speed_mps = gps_distance_and_speed(loc)
step_length_m = float(dist_m / steps_filtered) if steps_filtered > 0 else float("nan")


def fmt_float(x, nd=2):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{nd}f}"

with colA:
    st.subheader("Tulokset")
    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Askelmäärä (suodatettu)", f"{steps_filtered:d}")
    m2.metric("Askelmäärä (Fourier/PSD)", f"{steps_fft:d}")
    m3.metric("Keskinopeus (GPS)", f"{fmt_float(avg_speed_mps, 2)} m/s", f"{fmt_float(avg_speed_mps*3.6, 2)} km/h" if np.isfinite(avg_speed_mps) else None)
    m4.metric("Matka (GPS)", f"{fmt_float(dist_m/1000, 3)} km", f"{fmt_float(dist_m, 1)} m" if np.isfinite(dist_m) else None)
    m5.metric("Askelpituus", f"{fmt_float(step_length_m, 3)} m", f"{fmt_float(step_length_m*100, 1)} cm" if np.isfinite(step_length_m) else None)

    st.caption(f"Näytteenottotaajuus (arvio): {fmt_float(fs, 2)} Hz. PSD-dominantti: {fmt_float(peak_f, 3)} Hz. Tulkittu askeltaajuus: {fmt_float(step_freq_hz, 3)} Hz.")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Suodatettu kiihtyvyysdata (askelten määritykseen)")
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t_acc, sig_f)
    plt.xlabel("Aika [s]")
    plt.ylabel("Suodatettu kiihtyvyys [m/s²]")
    plt.title(f"{component} band-pass {low:.1f}–{high:.1f} Hz")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Tehospektritiheys (PSD) valitulle komponentille")
    fig = plt.figure(figsize=(10, 4))
    plt.semilogy(f_psd, P_psd)
    plt.xlim(0, 10)
    plt.xlabel("Taajuus [Hz]")
    plt.ylabel("Tehospektritiheys")
    plt.title(f"PSD – {component}")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

st.divider()
st.subheader("Komponenttien vertailu (SNR-heuristiikka)")
st.dataframe(snr_df, use_container_width=True)

st.divider()
st.subheader("Reitti kartalla")
try:
    html = folium_map_html(loc)
    components.html(html, height=600, scrolling=False)
except Exception as e:
    st.warning(f"Kartan renderöinti epäonnistui: {e}")