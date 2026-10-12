
from pathlib import Path
from io import BytesIO
from datetime import datetime, date, time
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")

# ============================================================
# CAMPUS IQ — SMART CLASSROOM OCCUPANCY & ENERGY ADVISOR
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

for folder in (DATA, MODELS, REPORTS, REPORTS / "figures"):
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="CAMPUS IQ | Classroom Advisor",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- DESIGN ---------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

.stApp {
    background: #F5F8FC;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: #10243A;
}
[data-testid="stSidebar"] * {
    color: #F2F7FC;
}
.hero {
    background: linear-gradient(120deg,#10243A,#164B63,#087E8B);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    font-size: 2.3rem;
    margin: 8px 0;
    font-family: 'Space Grotesk', sans-serif;
}
.hero p {
    color: #E1F0F5;
    margin-bottom: 0;
}
.kicker {
    color: #7BE2D0;
    font-size: .76rem;
    letter-spacing: 2px;
    font-weight: 700;
}
.section {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #17365D;
    margin: 24px 0 10px;
}
.card {
    background: white;
    border: 1px solid #E0E8F0;
    border-radius: 14px;
    padding: 18px;
    min-height: 112px;
}
.card-label {
    color: #63758B;
    font-size: .85rem;
}
.card-value {
    color: #17365D;
    font-size: 1.65rem;
    font-weight: 700;
    margin: 5px 0;
}
.note {
    background: #EAF5F7;
    border-left: 4px solid #087E8B;
    padding: 13px;
    border-radius: 6px;
    color: #24485A;
}
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    border-radius: 9px;
    font-weight: 600;
    min-height: 42px;
}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = {
    "template": "plotly_white",
    "font": {"family": "DM Sans", "color": "#34465E"},
    "margin": {"l": 45, "r": 25, "t": 70, "b": 55},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
}

PAGES = [
    "Home",
    "Check a Classroom",
    "Occupancy Trends",
    "Visualization Explorer",
    "Timetable & Editor",
    "Reports & Downloads",
    "Technical Insights",
    "About CAMPUS IQ",
]


def hero(kicker, title, subtitle):
    st.markdown(
        f'<div class="hero"><div class="kicker">{kicker}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(
        f'<div class="section">{title}</div>',
        unsafe_allow_html=True,
    )


def card(label, value, explanation=""):
    st.markdown(
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div>'
        f'<div class="card-label">{explanation}</div></div>',
        unsafe_allow_html=True,
    )


def norm(value):
    import re
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def read_table(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def find_existing(paths):
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


# ============================================================
# DATA LOADING
# ============================================================

def normalize_dataset(df):
    if df.empty:
        return df

    df = df.copy()
    df.columns = [
        str(c).strip().replace("\ufeff", "")
        for c in df.columns
    ]

    aliases = {
        "datetime": "date",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "light": "Light",
        "co2": "CO2",
        "humidityratio": "HumidityRatio",
        "occupancy": "Occupancy",
    }

    rename = {
        col: aliases[norm(col)]
        for col in df.columns
        if norm(col) in aliases
    }
    df = df.rename(columns=rename)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = df["date"].dt.hour
        df["day_of_week"] = df["date"].dt.dayofweek

    for col in [
        "Temperature", "Humidity", "Light", "CO2",
        "HumidityRatio", "Occupancy", "hour", "day_of_week",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data
def load_datasets():
    candidates = {
        "Training data": [
            "datatraining (1).txt",
            "datatraining.txt",
        ],
        "Test data 1": [
            "datatest (1).txt",
            "datatest.txt",
        ],
        "Test data 2": [
            "datatest2 (1).txt",
            "datatest2.txt",
        ],
    }

    result = {}

    for label, filenames in candidates.items():
        path = find_existing([DATA / name for name in filenames])

        if path:
            df = normalize_dataset(read_table(path))
            if not df.empty:
                result[label] = df

    return result


def load_saved_table(filename):
    return read_table(REPORTS / filename)


DATASETS = load_datasets()

ALL_DATA = (
    pd.concat(
        [df.assign(Source=name) for name, df in DATASETS.items()],
        ignore_index=True,
    )
    if DATASETS else pd.DataFrame()
)

RESULTS = load_saved_table("combined_model_results.csv")
if RESULTS.empty:
    RESULTS = load_saved_table("best_model_summary.csv")
if RESULTS.empty:
    RESULTS = read_table(ROOT / "model_results.csv")

DIAGNOSTICS = load_saved_table("diagnostic_metrics.csv")
ROC_RESULTS = load_saved_table("roc_auc_results.csv")
FEATURES = load_saved_table("feature_importance_results.csv")
QUALITY = load_saved_table("data_quality_report.csv")
SENSOR_SUMMARY = load_saved_table("sensor_summary_statistics.csv")


def load_model():
    path = find_existing([
        MODELS / "occupancy_model_candidate.pkl",
        MODELS / "occupancy_model.pkl",
        MODELS / "timetable_occupancy_model.pkl",
    ])

    if path is None:
        return None, None

    try:
        return joblib.load(path), str(path)
    except Exception as exc:
        return None, f"{path}: {exc}"


MODEL, MODEL_SOURCE = load_model()


# ============================================================
# TIMETABLE: LOAD, CREATE, EDIT, SAVE
# ============================================================

TIMETABLE_PATH = DATA / "timetable.csv"


def normalize_timetable(df):
    if df.empty:
        return df

    df = df.copy()
    df.columns = [
        str(c).strip().replace("\ufeff", "")
        for c in df.columns
    ]

    aliases = {
        "room": "Room",
        "classroom": "Room",
        "day": "Day",
        "start": "Start_Time",
        "starttime": "Start_Time",
        "end": "End_Time",
        "endtime": "End_Time",
        "subject": "Subject",
        "course": "Subject",
        "faculty": "Faculty",
        "classscheduled": "Class_Scheduled",
        "scheduled": "Class_Scheduled",
    }

    rename = {}
    for col in df.columns:
        key = norm(col)
        if key in aliases:
            rename[col] = aliases[key]

    return df.rename(columns=rename)


def load_timetable():
    candidates = [
        TIMETABLE_PATH,
        ROOT / "timetable.csv",
    ]
    candidates.extend(sorted(DATA.glob("*timetable*.csv")))

    path = find_existing(candidates)

    if path:
        df = normalize_timetable(read_table(path))
        if not df.empty:
            return df, str(path)

    # Clearly marked starter entries. These are NOT an official timetable.
    sample = pd.DataFrame([
        {
            "Room": "A101", "Day": "Monday",
            "Start_Time": "09:00", "End_Time": "10:00",
            "Subject": "SAMPLE - replace this",
            "Faculty": "", "Class_Scheduled": True,
        },
        {
            "Room": "A101", "Day": "Monday",
            "Start_Time": "10:00", "End_Time": "11:00",
            "Subject": "SAMPLE - replace this",
            "Faculty": "", "Class_Scheduled": True,
        },
        {
            "Room": "A102", "Day": "Tuesday",
            "Start_Time": "11:00", "End_Time": "12:00",
            "Subject": "SAMPLE - replace this",
            "Faculty": "", "Class_Scheduled": True,
        },
    ])

    try:
        sample.to_csv(TIMETABLE_PATH, index=False)
    except Exception:
        pass

    return sample, "Generated sample timetable - not an official schedule"


def save_timetable(df):
    required = {"Room", "Day", "Start_Time", "End_Time"}

    if not required.issubset(df.columns):
        raise ValueError(
            "The timetable must contain Room, Day, Start_Time and End_Time."
        )

    cleaned = df.copy()

    for col in required:
        cleaned[col] = cleaned[col].fillna("").astype(str).str.strip()

    if (cleaned[list(required)] == "").any().any():
        raise ValueError("Room, day, start time and end time cannot be blank.")

    DATA.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(TIMETABLE_PATH, index=False)


def time_minutes(value):
    try:
        parsed = datetime.strptime(str(value).strip(), "%H:%M")
        return parsed.hour * 60 + parsed.minute
    except Exception:
        try:
            parsed = pd.to_datetime(str(value), errors="coerce")
            if pd.notna(parsed):
                return parsed.hour * 60 + parsed.minute
        except Exception:
            pass
    return None


def check_schedule(df, room, when):
    required = {"Room", "Day", "Start_Time", "End_Time"}

    if df.empty or not required.issubset(df.columns):
        return None, "The timetable is missing required columns."

    rows = df[
        df["Room"].astype(str).str.strip().str.casefold()
        == str(room).strip().casefold()
    ]

    rows = rows[
        rows["Day"].astype(str).str.strip().str.casefold()
        == when.strftime("%A").casefold()
    ]

    current = when.hour * 60 + when.minute

    for _, row in rows.iterrows():
        start = time_minutes(row["Start_Time"])
        end = time_minutes(row["End_Time"])

        if start is not None and end is not None and start <= current < end:
            return row, "A timetable entry matches this room and time."

    return None, "No matching class was found for this room and time."


TIMETABLE, TIMETABLE_SOURCE = load_timetable()


# ============================================================
# CHART HELPERS
# ============================================================

def occupancy_counts(df):
    if df.empty or "Occupancy" not in df.columns:
        return pd.DataFrame()

    return pd.DataFrame({
        "Occupancy": ["Not occupied", "Occupied"],
        "Records": [
            int((df["Occupancy"] == 0).sum()),
            int((df["Occupancy"] == 1).sum()),
        ],
    })


def occupancy_chart(df, title="Occupancy distribution"):
    counts = occupancy_counts(df)

    if counts.empty:
        return None

    fig = px.bar(
        counts,
        x="Occupancy",
        y="Records",
        text="Records",
        color="Occupancy",
        color_discrete_map={
            "Not occupied": "#087E8B",
            "Occupied": "#F4A261",
        },
        title=title,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**PLOT_LAYOUT, showlegend=False)
    fig.update_yaxes(title="Number of records", rangemode="tozero")
    return fig


def hourly_chart(df):
    if df.empty or "Occupancy" not in df.columns:
        return None

    work = df.copy()

    if "date" in work.columns:
        work["hour"] = pd.to_datetime(
            work["date"], errors="coerce"
        ).dt.hour

    if "hour" not in work.columns:
        return None

    work = work.dropna(subset=["hour", "Occupancy"])

    if work.empty:
        return None

    hourly = work.groupby("hour", as_index=False)["Occupancy"].mean()
    hourly["Occupied records (%)"] = hourly["Occupancy"] * 100

    fig = px.line(
        hourly,
        x="hour",
        y="Occupied records (%)",
        markers=True,
        title="Share of records labelled occupied by hour",
    )
    fig.update_traces(line_color="#087E8B", line_width=3)
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_xaxes(title="Hour of day", dtick=2)
    fig.update_yaxes(title="Occupied records (%)", range=[0, 100])
    return fig


def sensor_chart(df, feature):
    if feature not in df.columns:
        return None

    series = pd.to_numeric(df[feature], errors="coerce").dropna()

    if series.empty:
        return None

    if len(series) > 2500:
        step = max(1, len(series) // 2500)
        series = series.iloc[::step]

    plot_df = pd.DataFrame({
        "Sampled record": np.arange(1, len(series) + 1),
        feature: series.to_numpy(),
    })

    fig = px.line(
        plot_df,
        x="Sampled record",
        y=feature,
        title=f"{feature}: sensor pattern across sampled records",
    )
    fig.update_traces(line_color="#5271A8")
    fig.update_layout(**PLOT_LAYOUT)
    return fig


# ============================================================
# MODEL PREDICTION
# ============================================================

def build_prediction_features(model, readings, when):
    values_by_name = {
        "temperature": readings["Temperature"],
        "humidity": readings["Humidity"],
        "light": readings["Light"],
        "co2": readings["CO2"],
        "humidityratio": readings["HumidityRatio"],
        "hour": when.hour,
        "dayofweek": when.weekday(),
        "day": when.weekday(),
        "weekday": when.weekday(),
    }

    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = [
            "Temperature", "Humidity", "Light", "CO2",
            "HumidityRatio", "hour", "day_of_week",
        ]

    row = {}

    for feature in features:
        key = norm(feature)

        if key not in values_by_name:
            raise ValueError(
                f"The saved model expects '{feature}', but this form "
                "does not supply that feature. The model and form need "
                "to use the same training features."
            )

        row[feature] = values_by_name[key]

    return pd.DataFrame([row], columns=features)


def predict_occupancy(model, readings, when):
    X = build_prediction_features(model, readings, when)
    prediction = model.predict(X)[0]
    confidence = None

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X)[0]
            classes = list(model.classes_)
            if prediction in classes:
                confidence = float(
                    probabilities[classes.index(prediction)]
                ) * 100
        except Exception:
            pass

    return int(prediction), confidence


# ============================================================
# REPORT BUILDERS
# ============================================================

def build_excel(tables):
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    workbook = Workbook()
    workbook.remove(workbook.active)

    for name, df in tables.items():
        if df is None or df.empty:
            continue

        sheet = workbook.create_sheet(str(name)[:31])

        for row in dataframe_to_rows(df, index=False, header=True):
            sheet.append(row)

        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

        for column in sheet.columns:
            width = min(
                max(len(str(cell.value or "")) for cell in column) + 2,
                38,
            )
            sheet.column_dimensions[
                column[0].column_letter
            ].width = max(12, width)

    output = BytesIO()
    workbook.save(output)
    output.seek(0)
    return output.getvalue()


def collect_user_report_selections():
    """Return a complete eight-module report inventory, including defaults and user-specific values."""
    field_groups = {
        "Home": [
            ("Module purpose", "Overview of available records and occupancy labels"),
            ("Dataset choices available", ", ".join(DATASETS.keys()) or "No datasets loaded"),
            ("Selected dataset", st.session_state.get("home_dataset", "Training data (default if available)")),
        ],
        "Check a Classroom": [
            ("Module purpose", "Manual sensor inputs, occupancy estimate, and timetable comparison"),
            ("Classroom label entered", st.session_state.get("classroom_room", "Not entered")),
            ("Temperature (°C)", st.session_state.get("classroom_temperature", 23.0)),
            ("Humidity (%)", st.session_state.get("classroom_humidity", 27.0)),
            ("Light reading", st.session_state.get("classroom_light", 450.0)),
            ("CO₂ reading (ppm)", st.session_state.get("classroom_co2", 600.0)),
            ("Humidity ratio", st.session_state.get("classroom_humidity_ratio", 0.0045)),
            ("Date", st.session_state.get("classroom_date", date.today())),
            ("Time", st.session_state.get("classroom_time", time(10, 30))),
            ("Latest result", st.session_state.get("last_estimate", "No prediction generated yet")),
        ],
        "Occupancy Trends": [
            ("Module purpose", "Occupancy distribution, hourly patterns, and sensor trends"),
            ("Selected dataset", st.session_state.get("trends_dataset", "First available dataset (default)")),
            ("Selected sensor", st.session_state.get("trends_sensor", "First available sensor (default)")),
        ],
        "Visualization Explorer": [
            ("Module purpose", "Interactive charts and record exploration"),
            ("Selected dataset", st.session_state.get("viz_dataset", "First available dataset (default)")),
            ("Question explored", st.session_state.get("viz_question", "Occupancy overview (default)")),
            ("Chart type", st.session_state.get("viz_chart_type", "First available style (default)")),
            ("Occupancy filters", st.session_state.get("viz_labels", [0, 1])),
            ("Sensor over time", st.session_state.get("viz_time_feature", "Default when applicable")),
            ("Sensor distribution", st.session_state.get("viz_distribution_feature", "Default when applicable")),
            ("Sensor comparison", st.session_state.get("viz_compare_feature", "Default when applicable")),
            ("Horizontal axis", st.session_state.get("viz_scatter_x", "Default when applicable")),
            ("Vertical axis", st.session_state.get("viz_scatter_y", "Default when applicable")),
        ],
        "Timetable & Editor": [
            ("Module purpose", "View, edit, save, download, and check timetable entries"),
            ("Timetable source", TIMETABLE_SOURCE),
            ("Selected room", st.session_state.get("schedule_room", "First available room (default)")),
            ("Selected day", st.session_state.get("schedule_day", "Monday (default)")),
            ("Selected time", st.session_state.get("schedule_check_time", time(10, 30))),
            ("Latest timetable check result", st.session_state.get("last_schedule_check", "No room/time check recorded yet")),
            ("Edited timetable", "See timetable preview table included in the report"),
        ],
        "Reports & Downloads": [
            ("Module purpose", "Export combined PDF, Excel analysis, and CSV data"),
            ("PDF contents", "All eight module summaries, session selections/results, available charts and tables, dataset/model/timetable context"),
        ],
        "Technical Insights": [
            ("Module purpose", "Saved model evaluation, metrics, ROC-AUC, feature importance, data quality, and sensor statistics"),
            ("Selected comparison metric", st.session_state.get("technical_metric", "First available numeric metric (default)")),
            ("Model source", MODEL_SOURCE or "No saved model loaded"),
        ],
        "About CAMPUS IQ": [
            ("Module purpose", "Project purpose, capabilities, limitations, and future improvements"),
            ("Key limitation", "Prototype only; not live occupancy, equipment status, or electricity measurement"),
        ],
    }
    rows = []
    changed_fields = set(st.session_state.get("campus_iq_user_changed_fields", []))
    field_to_key = {
        "Selected dataset": {"home_dataset", "trends_dataset", "viz_dataset"},
        "Selected sensor": {"trends_sensor"},
        "Question explored": {"viz_question"},
        "Chart type": {"viz_chart_type"},
        "Occupancy filters": {"viz_labels"},
        "Sensor over time": {"viz_time_feature"},
        "Sensor distribution": {"viz_distribution_feature"},
        "Sensor comparison": {"viz_compare_feature"},
        "Horizontal axis": {"viz_scatter_x"},
        "Vertical axis": {"viz_scatter_y"},
        "Classroom label entered": {"classroom_room"},
        "Temperature (°C)": {"classroom_temperature"},
        "Humidity (%)": {"classroom_humidity"},
        "Light reading": {"classroom_light"},
        "CO₂ reading (ppm)": {"classroom_co2"},
        "Humidity ratio": {"classroom_humidity_ratio"},
        "Date": {"classroom_date"},
        "Time": {"classroom_time", "schedule_check_time"},
        "Selected room": {"schedule_room"},
        "Selected day": {"schedule_day"},
        "Selected comparison metric": {"technical_metric"},
    }
    for module, fields in field_groups.items():
        for label, value in fields:
            keys = field_to_key.get(label, set())
            if keys and not (keys & changed_fields):
                display_value = "Not selected" if label in {
                    "Selected dataset", "Selected sensor", "Question explored", "Chart type",
                    "Occupancy filters", "Sensor over time", "Sensor distribution", "Sensor comparison",
                    "Horizontal axis", "Vertical axis", "Selected room", "Selected day",
                    "Selected comparison metric"
                } else "Not entered"
            else:
                display_value = value
            if isinstance(display_value, (dict, list, tuple)):
                display_value = str(display_value)
            elif isinstance(display_value, (datetime, date, time)):
                display_value = display_value.isoformat()
            rows.append({"Section": module, "Selection / information": label, "Recorded value / explanation": str(display_value)})
    return pd.DataFrame(rows, columns=["Section", "Selection / information", "Recorded value / explanation"])


def build_pdf(datasets, results, quality, features, timetable, user_selections=None, last_estimate=None, report_visualization=None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    )
    import matplotlib.pyplot as plt

    output = BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=38,
        rightMargin=38,
        topMargin=38,
        bottomMargin=38,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        alignment=1,
        textColor=colors.HexColor("#52657A"),
        spaceAfter=16,
    ))

    story = [
        Paragraph("CAMPUS IQ", styles["Title"]),
        Paragraph(
            "Smart Classroom Occupancy &amp; Energy Advisor",
            styles["ReportSubtitle"],
        ),
        Paragraph("Readable project report", styles["Heading1"]),
        Paragraph(
            "Generated: " + datetime.now().strftime("%d %B %Y, %H:%M"),
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph("Executive summary", styles["Heading1"]),
        Paragraph(
            "CAMPUS IQ estimates occupancy labels from environmental sensor "
            "readings and helps users review those estimates alongside a timetable. "
            "It is a prototype decision-support tool, not a live building-control system.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("How to understand the graphs", styles["Heading1"]),
        Paragraph(
            "Occupancy charts count records labelled occupied or not occupied. "
            "Hourly charts show the share of records labelled occupied at each hour. "
            "Sensor charts show how recorded measurements vary. These patterns do "
            "not prove cause and effect.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("Important limitations", styles["Heading1"]),
        Paragraph(
            "The supplied public dataset is not a live campus sensor feed. Its "
            "records are not automatically linked to actual college rooms. The "
            "prototype does not measure equipment ON/OFF status, electricity "
            "consumption or verified energy savings. Sample timetable entries "
            "must be replaced with an authorised timetable.",
            styles["BodyText"],
        ),
        Spacer(1, 10),
    ]

    def add_table(title, frame, limit=30):
        from reportlab.platypus import Paragraph
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_LEFT

        if frame is None or frame.empty:
            return

        story.append(Paragraph(title, styles["Heading2"]))

        frame = frame.head(limit).copy()

        # Keep tables within the printable A4 page width.
        page_width = A4[0] - doc.leftMargin - doc.rightMargin

        cell_style = ParagraphStyle(
            "ReportTableCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7,
            leading=8.5,
            alignment=TA_LEFT,
            wordWrap="CJK",
        )

        header_style = ParagraphStyle(
            "ReportTableHeader",
            parent=cell_style,
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )

        def format_value(value, column):
            # Report cells can occasionally contain pandas Series/arrays or
            # other nested objects. Never pass those objects directly to
            # ReportLab: they can produce ambiguous truth checks or enormous
            # table rows that cannot fit on a PDF page.
            if isinstance(value, pd.Series):
                if len(value) == 0:
                    return "Not available"
                preview = value.astype(str).head(8).tolist()
                suffix = " …" if len(value) > 8 else ""
                value = ", ".join(preview) + suffix
            elif isinstance(value, (list, tuple, set, np.ndarray)):
                items = list(value)
                suffix = " …" if len(items) > 8 else ""
                value = ", ".join(str(v) for v in items[:8]) + suffix
            elif isinstance(value, dict):
                value = "; ".join(f"{k}: {v}" for k, v in list(value.items())[:8])

            try:
                missing = pd.isna(value)
                if isinstance(missing, (bool, np.bool_)) and missing:
                    return "Not available"
            except (TypeError, ValueError):
                pass

            value = str(value)
            # Keep a single PDF cell at a manageable height. Full supporting
            # data remains available in the app's CSV/Excel downloads.
            max_cell_chars = 600
            if len(value) > max_cell_chars:
                value = value[:max_cell_chars] + " … [truncated in PDF; see supporting data]"

            # Show evaluation metrics as percentages.
            metric_names = {
                "accuracy", "precision", "recall", "f1",
                "f1-score", "f1_score", "roc_auc", "roc-auc"
            }

            if column.strip().lower() in metric_names:
                try:
                    number = float(value)
                    if 0 <= number <= 1:
                        return f"{number * 100:.2f}%"
                except (TypeError, ValueError):
                    pass

            if isinstance(value, float):
                return f"{value:.3f}"

            return str(value)

        columns = list(frame.columns)

        rows = [[
            Paragraph(str(column), header_style)
            for column in columns
        ]]

        for _, record in frame.iterrows():
            rows.append([
                Paragraph(
                    format_value(record[column], str(column))
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"),
                    cell_style,
                )
                for column in columns
            ])

        # Allocate width based on column names, while ensuring the
        # sum of all widths stays within the printable page width.
        weights = [
            max(8, min(22, len(str(column))))
            for column in columns
        ]
        total_weight = sum(weights)

        column_widths = [
            page_width * weight / total_weight
            for weight in weights
        ]

        table = Table(
            rows,
            colWidths=column_widths,
            repeatRows=1,
            hAlign="LEFT",
            splitByRow=1,
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#12324A")),
            ("GRID", (0, 0), (-1, -1),
             0.35, colors.HexColor("#D6E0EA")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F1F6FA")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))

        story.extend([table, Spacer(1, 10)])
    
        # User selections captured from the app
    if not isinstance(user_selections, pd.DataFrame):
        user_selections = pd.DataFrame(
            columns=["Section", "Selection", "User choice"]
        )

    if not user_selections.empty:
        story.append(
            Paragraph(
                "Your Selections Across CAMPUS IQ",
                styles["Heading1"],
            )
        )
        story.append(
            Paragraph(
                "This section summarizes the choices currently recorded "
                "in your app session. It does not include past choices "
                "that were never recorded.",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 6))
        add_table("Selections by Page", user_selections, 100)
    
    if last_estimate:
        estimate_df = pd.DataFrame(
            list(last_estimate.items()),
            columns=["Detail", "Your Result"],
        )

        story.append(
            Paragraph(
                "Your Latest Classroom Prediction",
                styles["Heading1"],
            )
        )

        add_table(
            "Prediction Details",
            estimate_df,
            20,
        )
    
    if report_visualization is None:
        report_visualization = {}
    if report_visualization:
        story.append(
            Paragraph(
                "Your Selected Visualization",
                styles["Heading1"],
            )
        )

        visualization_rows = [
            ["Setting", "Your choice"],
            ["Dataset", report_visualization.get("dataset", "Not recorded")],
            ["Question", report_visualization.get("question", "Not recorded")],
            ["Chart type", report_visualization.get("chart_type", "Not recorded")],
            [
                "Occupancy filters",
                str(report_visualization.get("filters", [])),
            ],
            [
                "Records displayed",
                str(report_visualization.get("records_displayed", "Not recorded")),
            ],
        ]

        add_table(
            "Visualization Details",
            pd.DataFrame(
                visualization_rows[1:],
                columns=visualization_rows[0],
            ),
        )
        
        chart_image = report_visualization.get("image")

        if chart_image:
            from reportlab.platypus import Image
            story.append(Spacer(1, 8))
            story.append(
                Image(
                BytesIO(chart_image),
                width=500,
                height=300,
                kind="proportional",
                )
                )
            story.append(Spacer(1, 10))


    if datasets:
        story.append(Paragraph("Dataset overview", styles["Heading1"]))

        overview = pd.DataFrame([
            {
                "Dataset": name,
                "Records": len(df),
                "Occupied": int((df["Occupancy"] == 1).sum())
                    if "Occupancy" in df else "Unavailable",
                "Not occupied": int((df["Occupancy"] == 0).sum())
                    if "Occupancy" in df else "Unavailable",
            }
            for name, df in datasets.items()
        ])
        add_table("Dataset summary", overview)

        for name, df in datasets.items():
            counts = occupancy_counts(df)

            if counts.empty:
                continue

            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.bar(
                counts["Occupancy"],
                counts["Records"],
                color=["#087E8B", "#F4A261"],
            )
            ax.set_title(f"{name}: occupancy distribution")
            ax.set_ylabel("Number of records")
            ax.tick_params(axis="x", rotation=0)
            fig.tight_layout()

            image_buffer = BytesIO()
            fig.savefig(image_buffer, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            image_buffer.seek(0)

            story.append(Paragraph(
                f"Occupancy distribution — {name}",
                styles["Heading2"],
            ))
            story.append(Image(
                image_buffer,
                width=6.4 * inch,
                height=3.0 * inch,
            ))
            story.append(Paragraph(
                "Interpretation: this graph compares the number of records "
                "in each occupancy class for the selected dataset. These are "
                "dataset labels, not a live count of people in campus rooms.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 5))

            # Always include the default Home/Occupancy Trends hourly graph
            # when the dataset contains usable time information.
            try:
                hour_work = df.copy()
                if "date" in hour_work.columns and "Occupancy" in hour_work.columns:
                    hour_work["_report_hour"] = pd.to_datetime(hour_work["date"], errors="coerce").dt.hour
                elif "hour" in hour_work.columns:
                    hour_work["_report_hour"] = pd.to_numeric(hour_work["hour"], errors="coerce")
                else:
                    hour_work["_report_hour"] = np.nan
                hour_work = hour_work.dropna(subset=["_report_hour", "Occupancy"])
                if not hour_work.empty:
                    hourly_values = hour_work.groupby("_report_hour")["Occupancy"].mean().sort_index() * 100
                    hourly_fig, hourly_ax = plt.subplots(figsize=(7, 3.5))
                    hourly_ax.plot(hourly_values.index, hourly_values.values, marker="o", linewidth=2, color="#087E8B")
                    hourly_ax.set_title(f"{name}: share of records labelled occupied by hour")
                    hourly_ax.set_xlabel("Hour of day")
                    hourly_ax.set_ylabel("Occupied records (%)")
                    hourly_ax.set_ylim(0, 100)
                    hourly_ax.set_xticks(range(0, 24, 2))
                    hourly_ax.grid(axis="y", alpha=0.25)
                    hourly_fig.tight_layout()
                    hourly_png = BytesIO()
                    hourly_fig.savefig(hourly_png, format="png", dpi=150, bbox_inches="tight")
                    plt.close(hourly_fig)
                    hourly_png.seek(0)
                    story.append(Paragraph(f"Hourly occupancy pattern — {name}", styles["Heading2"]))
                    story.append(Image(hourly_png, width=6.4 * inch, height=3.0 * inch, kind="proportional"))
                    story.append(Paragraph(
                        "Interpretation: each point shows the percentage of records labelled occupied at that hour. "
                        "A missing or sparse hour reflects the supplied data, not necessarily a real-world pattern.",
                        styles["BodyText"],
                    ))
                    story.append(Spacer(1, 5))
            except Exception:
                pass

            # Include the default sensor trend shown by Occupancy Trends.
            try:
                default_sensor = next((c for c in ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"] if c in df.columns), None)
                if default_sensor:
                    sensor_values = pd.to_numeric(df[default_sensor], errors="coerce").dropna()
                    if not sensor_values.empty:
                        if len(sensor_values) > 2500:
                            sensor_values = sensor_values.iloc[::max(1, len(sensor_values) // 2500)]
                        sensor_fig, sensor_ax = plt.subplots(figsize=(7, 3.5))
                        sensor_ax.plot(np.arange(len(sensor_values)), sensor_values.values, linewidth=1.2, color="#5575A8")
                        sensor_ax.set_title(f"{name}: {default_sensor} sensor pattern")
                        sensor_ax.set_xlabel("Sampled record")
                        sensor_ax.set_ylabel(default_sensor)
                        sensor_ax.grid(axis="y", alpha=0.25)
                        sensor_fig.tight_layout()
                        sensor_png = BytesIO()
                        sensor_fig.savefig(sensor_png, format="png", dpi=150, bbox_inches="tight")
                        plt.close(sensor_fig)
                        sensor_png.seek(0)
                        story.append(Paragraph(f"Default sensor trend — {name} ({default_sensor})", styles["Heading2"]))
                        story.append(Image(sensor_png, width=6.4 * inch, height=3.0 * inch, kind="proportional"))
                        story.append(Paragraph(
                            "Interpretation: this graph shows how the selected sensor's recorded values vary across sampled records. "
                            "It does not prove cause and effect.", styles["BodyText"]
                        ))
                        story.append(Spacer(1, 8))
            except Exception:
                pass

    # The default Visualization Explorer view must be present in every report,
    # even when the user has not opened that page during this session.
    if datasets:
        default_name = next(iter(datasets))
        default_df = datasets[default_name]
        default_feature = next((c for c in ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"] if c in default_df.columns), None)
        if default_feature and "Occupancy" in default_df.columns:
            try:
                story.append(Paragraph("Visualization Explorer — default view", styles["Heading1"]))
                story.append(Paragraph(
                    f"Default dataset: {default_name}. Default example: distribution of {default_feature} readings, "
                    "shown separately for occupied and not-occupied labels. This standard view is included in every "
                    "report; the user's selected visualization, when recorded, appears separately below.",
                    styles["BodyText"]
                ))
                default_fig, default_ax = plt.subplots(figsize=(7.2, 3.7))
                for label, color, label_name in [(0, "#087E8B", "Not occupied"), (1, "#F4A261", "Occupied")]:
                    vals = pd.to_numeric(default_df.loc[default_df["Occupancy"] == label, default_feature], errors="coerce").dropna()
                    if not vals.empty:
                        default_ax.hist(vals, bins=25, alpha=0.55, color=color, label=label_name)
                default_ax.set_title(f"Distribution of {default_feature} readings")
                default_ax.set_xlabel(default_feature)
                default_ax.set_ylabel("Number of records")
                default_ax.legend()
                default_fig.tight_layout()
                default_png = BytesIO()
                default_fig.savefig(default_png, format="png", dpi=150, bbox_inches="tight")
                plt.close(default_fig)
                default_png.seek(0)
                story.append(Image(default_png, width=6.4 * inch, height=3.0 * inch, kind="proportional"))
                story.append(Paragraph(
                    "How to read it: the bars show how frequently sensor values occur in the two labelled groups. "
                    "Overlap means values in that range appear in both groups; this chart alone does not prove that "
                    "the sensor causes occupancy.", styles["BodyText"]
                ))
                story.append(Spacer(1, 10))
            except Exception:
                pass

    if results is not None and not results.empty:
        story.append(Paragraph("Saved model evaluation", styles["Heading1"]))
        story.append(Paragraph(
            "Accuracy measures overall correctness. Precision measures how "
            "often positive predictions are correct. Recall measures how many "
            "actual positive cases were identified. F1 balances precision and "
            "recall. Interpret these metrics together.",
            styles["BodyText"],
        ))
        add_table("Saved model comparison", results, 40)

    if quality is not None and not quality.empty:
        story.append(Paragraph("Data quality", styles["Heading1"]))
        add_table("Saved data quality results", quality)

    if features is not None and not features.empty:
        story.append(Paragraph("Understanding model inputs", styles["Heading1"]))
        add_table("Saved feature-importance results", features)
        story.append(Paragraph(
            "Feature importance describes how inputs contribute to a model "
            "under the saved analysis method. It does not establish causation.",
            styles["BodyText"],
        ))

    if timetable is not None and not timetable.empty:
        story.append(Paragraph("Timetable context", styles["Heading1"]))
        add_table("Timetable preview", timetable, 500)

    story.append(Paragraph("Complete report across all eight modules", styles["Heading1"]))
    story.append(Paragraph(
        "Every module is represented below, including its purpose and information available by default. "
        "Selections and results are included when recorded in this app session. A prediction is shown only "
        "if the user actually generated one.", styles["BodyText"]
    ))
    module_descriptions = [
        ("Home", "Dataset overview, record counts, occupancy-label share, occupancy distribution and hourly pattern."),
        ("Check a Classroom", "Manual sensor inputs, estimated occupancy, model score when available, and timetable comparison."),
        ("Occupancy Trends", "Selected dataset's occupancy distribution, hourly occupancy pattern, selected sensor trend, and CSV export."),
        ("Visualization Explorer", "Selected question, chart style, filters, sensor/axis choices, graph explanation, and chart data when captured."),
        ("Timetable & Editor", "Timetable source, editable timetable contents, room/day/time check, and save/download workflow."),
        ("Reports & Downloads", "PDF, Excel, and CSV export options plus available dataset and model summaries."),
        ("Technical Insights", "Saved model comparison and available evaluation, ROC-AUC, feature-importance, data-quality, and sensor-statistics tables."),
        ("About CAMPUS IQ", "Project purpose, capabilities, limitations, and future improvements."),
    ]
    add_table("Module guide", pd.DataFrame(module_descriptions, columns=["Module", "Default information and outputs"]))

    # Include the About page's standard content in every report, regardless
    # of whether the user visited that page during this session.
    story.append(Paragraph("About CAMPUS IQ", styles["Heading1"]))
    story.append(Paragraph(
        "CAMPUS IQ is a machine-learning prototype for exploring occupancy labels "
        "from environmental sensor readings, estimating whether manually entered "
        "readings resemble occupied conditions, and reviewing those estimates "
        "alongside timetable entries.", styles["BodyText"]
    ))
    story.append(Paragraph("What the prototype can do", styles["Heading2"]))
    story.append(Paragraph(
        "• Estimate occupied or not-occupied labels from supported sensor inputs.<br/>"
        "• Explore patterns in the supplied datasets using charts and summaries.<br/>"
        "• Compare an estimate with timetable entries for manual review.<br/>"
        "• View, edit, save and download timetable entries.<br/>"
        "• Export readable PDF reports, analysis workbooks and CSV data.",
        styles["BodyText"]
    ))
    story.append(Paragraph("What it cannot confirm", styles["Heading2"]))
    story.append(Paragraph(
        "The supplied public dataset is not a live campus sensor feed. The prototype "
        "cannot independently confirm live room occupancy, lights or air-conditioning "
        "status, actual electricity consumption, or verified energy savings. Sample "
        "timetable entries are not an authorised college schedule and must be replaced "
        "before relying on schedule comparisons.", styles["BodyText"]
    ))
    story.append(Paragraph("Future improvements", styles["Heading2"]))
    story.append(Paragraph(
        "Potential improvements include authorised live-sensor integration, verified "
        "timetable integration, equipment-status measurements, electricity-meter data, "
        "and validation using real campus observations.", styles["BodyText"]
    ))
    story.append(Spacer(1, 10))

    # Freeze the report inputs for this build. The PDF therefore reflects the
    # selections/results present when the download is generated, even if the user
    # later navigates back and changes a page.
    artifacts = {
        name: [dict(item) for item in items]
        for name, items in (st.session_state.get("campus_iq_report_artifacts", {}) or {}).items()
    }
    report_visualization = dict(report_visualization or {})
    for module_name in PAGES:
        story.append(Paragraph(module_name, styles["Heading2"]))
        module_items = artifacts.get(module_name, [])
        changed_fields = set(st.session_state.get("campus_iq_user_changed_fields", []))
        module_widget_keys = {
            "Home": {"home_dataset"},
            "Check a Classroom": {"classroom_room", "classroom_temperature", "classroom_humidity", "classroom_light", "classroom_co2", "classroom_humidity_ratio", "classroom_date", "classroom_time"},
            "Occupancy Trends": {"trends_dataset", "trends_sensor"},
            "Visualization Explorer": {"viz_dataset", "viz_question", "viz_chart_type", "viz_labels", "viz_time_feature", "viz_distribution_feature", "viz_compare_feature", "viz_scatter_x", "viz_scatter_y"},
            "Timetable & Editor": {"schedule_room", "schedule_day", "schedule_check_time"},
            "Reports & Downloads": set(),
            "Technical Insights": {"technical_metric"},
            "About CAMPUS IQ": set(),
        }
        user_made_selection = bool(module_widget_keys.get(module_name, set()) & changed_fields)
        story.append(Paragraph(
            "User selection status: Selected / entered" if user_made_selection else "User selection status: Not selected",
            styles["BodyText"]
        ))
        if not module_items:
            story.append(Paragraph(
                "No interaction-specific graph, table, or result was recorded for this module. "
                "Any item not selected by the user is marked Not selected; any action not performed is not presented as a generated result.",
                styles["BodyText"]
            ))
            story.append(Spacer(1, 6))
            continue
        for item in module_items:
            story.append(Paragraph(str(item.get("title", item.get("kind", "Captured item"))), styles["Heading3"]))
            if item.get("explanation"):
                story.append(Paragraph(str(item["explanation"]), styles["BodyText"]))
            if item.get("kind") == "table":
                try:
                    frame = pd.DataFrame(item.get("data", []), columns=item.get("columns"))
                    add_table(str(item.get("title", "Displayed table")), frame, 500)
                except Exception:
                    pass
            elif item.get("kind") == "summary":
                summary = item.get("data", {})
                if isinstance(summary, dict):
                    add_table(str(item.get("title", "Module summary")), pd.DataFrame([{"Detail": k, "Value": str(v)} for k, v in summary.items()]), 100)
            elif item.get("kind") == "chart":
                # Reports should show the actual graph, not a table of Plotly's
                # internal JSON/binary representation. Prefer the exact image
                # captured from the app; if unavailable, redraw the captured
                # chart data in Matplotlib for a readable PDF fallback.
                rendered = False
                if item.get("image"):
                    try:
                        image_buffer = BytesIO(item["image"])
                        story.append(Image(
                            image_buffer, width=6.2 * inch,
                            height=3.7 * inch, kind="proportional"
                        ))
                        rendered = True
                    except Exception:
                        rendered = False

                if not rendered and item.get("plotly_json"):
                    try:
                        import json
                        import base64

                        spec = json.loads(item["plotly_json"])

                        def report_plot_values(values):
                            # Plotly JSON may encode NumPy arrays as
                            # {"dtype": "...", "bdata": "..."}.
                            if isinstance(values, dict) and "bdata" in values:
                                dtype = np.dtype(values.get("dtype", "f8"))
                                raw = base64.b64decode(values["bdata"])
                                return np.frombuffer(raw, dtype=dtype).tolist()
                            if values is None:
                                return []
                            if isinstance(values, (list, tuple, np.ndarray)):
                                return list(values)
                            return [values]

                        plot_fig, plot_ax = plt.subplots(figsize=(8.2, 4.2))
                        any_plotted = False
                        for trace_no, trace in enumerate(spec.get("data", [])):
                            trace_type = trace.get("type", "scatter")
                            xs = report_plot_values(trace.get("x"))
                            ys = report_plot_values(trace.get("y"))
                            name = trace.get("name") or f"Series {trace_no + 1}"
                            if trace_type == "bar" and ys:
                                labels = [str(x) for x in xs] if xs else [str(i) for i in range(len(ys))]
                                plot_ax.bar(labels[:len(ys)], ys, label=name if len(spec.get("data", [])) > 1 else None)
                                any_plotted = True
                            elif trace_type == "scatter" and ys:
                                if not xs:
                                    xs = list(range(len(ys)))
                                mode = trace.get("mode", "lines")
                                if "markers" in mode and "lines" not in mode:
                                    plot_ax.scatter(xs[:len(ys)], ys, label=name)
                                else:
                                    plot_ax.plot(xs[:len(ys)], ys, marker="o" if "markers" in mode else None, label=name)
                                any_plotted = True
                            elif trace_type == "pie" and ys:
                                labels = trace.get("labels") or xs or [str(i) for i in range(len(ys))]
                                plot_ax.clear()
                                plot_ax.pie(ys, labels=labels, autopct="%1.1f%%")
                                any_plotted = True
                            elif trace_type == "box":
                                # Plotly box traces may contain raw y-values, or
                                # precomputed quartiles/fences. Handle both so a
                                # user-selected Box plot is not silently omitted.
                                if ys:
                                    if xs and len(xs) == len(ys):
                                        grouped = {}
                                        for x_value, y_value in zip(xs, ys):
                                            grouped.setdefault(str(x_value), []).append(y_value)
                                        group_labels = list(grouped.keys())
                                        group_values = [grouped[label] for label in group_labels]
                                        plot_ax.boxplot(group_values, labels=group_labels)
                                    else:
                                        plot_ax.boxplot(ys, labels=[name])
                                    any_plotted = True
                                elif all(key in trace for key in ("q1", "median", "q3")):
                                    q1_values = report_plot_values(trace.get("q1"))
                                    med_values = report_plot_values(trace.get("median"))
                                    q3_values = report_plot_values(trace.get("q3"))
                                    low_values = report_plot_values(trace.get("lowerfence")) or q1_values
                                    high_values = report_plot_values(trace.get("upperfence")) or q3_values
                                    count = min(len(q1_values), len(med_values), len(q3_values), len(low_values), len(high_values))
                                    if count:
                                        stats = [{
                                            "label": str(xs[i]) if i < len(xs) else (name if count == 1 else str(i + 1)),
                                            "q1": float(q1_values[i]),
                                            "med": float(med_values[i]),
                                            "q3": float(q3_values[i]),
                                            "whislo": float(low_values[i]),
                                            "whishi": float(high_values[i]),
                                            "fliers": [],
                                        } for i in range(count)]
                                        plot_ax.bxp(stats, showfliers=False)
                                        any_plotted = True

                        layout = spec.get("layout", {})
                        title_obj = layout.get("title", {})
                        chart_title = title_obj.get("text", "") if isinstance(title_obj, dict) else str(title_obj or "")
                        if chart_title:
                            plot_ax.set_title(chart_title)
                        x_title = layout.get("xaxis", {}).get("title", {})
                        y_title = layout.get("yaxis", {}).get("title", {})
                        if isinstance(x_title, dict):
                            x_title = x_title.get("text", "")
                        if isinstance(y_title, dict):
                            y_title = y_title.get("text", "")
                        if x_title:
                            plot_ax.set_xlabel(x_title)
                        if y_title:
                            plot_ax.set_ylabel(y_title)
                        if len(spec.get("data", [])) > 1:
                            plot_ax.legend(loc="best")
                        plot_ax.grid(axis="y", alpha=0.2)
                        plot_fig.tight_layout()

                        if any_plotted:
                            image_buffer = BytesIO()
                            plot_fig.savefig(image_buffer, format="png", dpi=160, bbox_inches="tight")
                            image_buffer.seek(0)
                            story.append(Image(
                                image_buffer, width=6.2 * inch,
                                height=3.7 * inch, kind="proportional"
                            ))
                            rendered = True
                        plt.close(plot_fig)
                    except Exception:
                        try:
                            plt.close("all")
                        except Exception:
                            pass

                if not rendered:
                    story.append(Paragraph(
                        "This chart was selected or displayed in the app, but its graph image "
                        "could not be recreated for this PDF export.",
                        styles["BodyText"]
                    ))
                if item.get("selection_status"):
                    story.append(Paragraph("Selection status: " + str(item["selection_status"]), styles["BodyText"]))
                story.append(Paragraph(
                    item.get("explanation") or
                    "This graph visualizes the data and options selected in the corresponding CAMPUS IQ module. "
                    "Read the axis labels and legend to understand the displayed comparison or pattern.",
                    styles["BodyText"]
                ))
                story.append(Spacer(1, 8))

    story.extend([
        Paragraph("Recommended next steps", styles["Heading1"]),
        Paragraph(
            "Verify unusual estimates manually. For actual campus deployment, "
            "connect authorised sensors, validate the model using campus data, "
            "use an approved timetable, and obtain equipment-status and "
            "electricity-meter measurements before making energy claims.",
            styles["BodyText"],
        ),
    ])

    doc.build(story)
    output.seek(0)
    return output.getvalue()

# ============================================================
# DURABLE WIDGET STATE + REPORT CAPTURE
# ============================================================

def _safe_report_value(value):
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple, set)):
        return [_safe_report_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_report_value(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _widget_default(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _widget_callback(widget_key, durable_key, kind=None):
    def _save():
        value = st.session_state.get(widget_key)
        if kind == "dataframe" and isinstance(value, pd.DataFrame):
            st.session_state[durable_key] = value.copy()
        else:
            st.session_state[durable_key] = _safe_report_value(value)
        # Report-only audit marker: distinguishes an actual user choice/input
        # from a widget's initial default. It does not alter the page UI.
        changed = st.session_state.setdefault("campus_iq_user_changed_fields", [])
        if durable_key not in changed:
            changed.append(durable_key)
    return _save


def _persistent_selectbox(label, options, key, **kwargs):
    options = list(options)
    default_index = kwargs.pop("index", 0)
    default = options[default_index] if options and 0 <= default_index < len(options) else None
    saved = _widget_default(key, default)
    if options and saved not in options:
        saved = default
        st.session_state[key] = saved
    widget_key = f"_campus_widget_{key}"
    if widget_key not in st.session_state or st.session_state.get(widget_key) not in options:
        st.session_state[widget_key] = saved
    selected = st.selectbox(label, options, index=options.index(st.session_state[widget_key]) if options and st.session_state[widget_key] in options else 0, key=widget_key, on_change=_widget_callback(widget_key, key), **kwargs)
    st.session_state[key] = selected
    return selected


def _persistent_multiselect(label, options, key, **kwargs):
    options = list(options)
    default = kwargs.pop("default", [])
    saved = _widget_default(key, default)
    if not isinstance(saved, list):
        saved = list(saved) if isinstance(saved, (tuple, set)) else default
    saved = [v for v in saved if v in options]
    st.session_state[key] = saved
    widget_key = f"_campus_widget_{key}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = saved
    else:
        st.session_state[widget_key] = [v for v in st.session_state[widget_key] if v in options]
    selected = st.multiselect(label, options, default=st.session_state[widget_key], key=widget_key, on_change=_widget_callback(widget_key, key), **kwargs)
    st.session_state[key] = list(selected)
    return selected


def _persistent_text_input(label, key, **kwargs):
    default = kwargs.pop("value", "")
    saved = _widget_default(key, default)
    widget_key = f"_campus_widget_{key}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = saved
    value = st.text_input(label, key=widget_key, on_change=_widget_callback(widget_key, key), **kwargs)
    st.session_state[key] = value
    return value


def _persistent_number_input(label, key, **kwargs):
    default = kwargs.pop("value", 0)
    saved = _widget_default(key, default)
    try:
        saved = min(max(saved, kwargs.get("min_value", saved)), kwargs.get("max_value", saved))
    except Exception:
        saved = default
    widget_key = f"_campus_widget_{key}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = saved
    value = st.number_input(label, key=widget_key, on_change=_widget_callback(widget_key, key), **kwargs)
    st.session_state[key] = value
    return value


def _persistent_date_input(label, key, **kwargs):
    default = kwargs.pop("value", date.today())
    saved = _widget_default(key, default)
    if isinstance(saved, str):
        try: saved = date.fromisoformat(saved)
        except Exception: saved = default
    widget_key = f"_campus_widget_{key}"
    if widget_key not in st.session_state: st.session_state[widget_key] = saved
    value = st.date_input(label, key=widget_key, on_change=_widget_callback(widget_key, key), **kwargs)
    st.session_state[key] = value.isoformat() if isinstance(value, (date, datetime)) else str(value)
    return value


def _persistent_time_input(label, key, **kwargs):
    default = kwargs.pop("value", time(10, 30))
    saved = _widget_default(key, default)
    if isinstance(saved, str):
        try: saved = time.fromisoformat(saved)
        except Exception: saved = default
    widget_key = f"_campus_widget_{key}"
    if widget_key not in st.session_state: st.session_state[widget_key] = saved
    value = st.time_input(label, key=widget_key, on_change=_widget_callback(widget_key, key), **kwargs)
    st.session_state[key] = value.isoformat() if isinstance(value, time) else str(value)
    return value


def _record_report_artifact(page_name, artifact):
    pages = st.session_state.setdefault("campus_iq_report_artifacts", {})
    items = pages.setdefault(page_name, [])
    identity = artifact.get("identity") or artifact.get("title") or artifact.get("kind")
    for i, old in enumerate(items):
        if (old.get("identity") or old.get("title") or old.get("kind")) == identity:
            items[i] = artifact
            return
    items.append(artifact)


def _show_plotly_chart(fig, *args, **kwargs):
    page_name = st.session_state.get("main_page", "Home")
    try:
        title = str(getattr(getattr(fig.layout, "title", None), "text", None) or "Chart")
        artifact = {
            "kind": "chart", "title": title, "identity": title,
            "plotly_json": fig.to_json(), "image": None,
            "selection_status": "Selected / displayed" if (
                (page_name == "Visualization Explorer" and set(st.session_state.get("campus_iq_user_changed_fields", [])) & {
                    "viz_dataset", "viz_question", "viz_chart_type", "viz_labels", "viz_time_feature",
                    "viz_distribution_feature", "viz_compare_feature", "viz_scatter_x", "viz_scatter_y"
                }) or (page_name == "Occupancy Trends" and set(st.session_state.get("campus_iq_user_changed_fields", [])) & {"trends_dataset", "trends_sensor"})
            ) else "Default page chart / no explicit chart selection recorded",
            "explanation": "Chart shown in this module. Its underlying plotted data is included when available.",
        }
        try:
            artifact["image"] = fig.to_image(format="png", width=1000, height=600)
        except Exception:
            pass
        _record_report_artifact(page_name, artifact)
    except Exception:
        pass
    return st.plotly_chart(fig, *args, **kwargs)


def _show_dataframe(data, *args, **kwargs):
    try:
        frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        if not frame.empty:
            page_name = st.session_state.get("main_page", "Home")
            _record_report_artifact(page_name, {
                "kind": "table", "title": "Displayed table", "identity": "table:" + "|".join(map(str, frame.columns)),
                "data": frame.head(500).to_dict(orient="records"), "columns": [str(c) for c in frame.columns],
                "explanation": "Displayed table; up to 500 rows are included in the PDF.",
            })
    except Exception:
        pass
    return st.dataframe(data, *args, **kwargs)


# ============================================================
# REPORT ACTIVITY LOG
# ============================================================

if "campus_iq_activity_log" not in st.session_state:
    st.session_state["campus_iq_activity_log"] = []

current_page = st.session_state.get("main_page", "Home")
activity_log = st.session_state["campus_iq_activity_log"]

if not activity_log or activity_log[-1].get("Page") != current_page:
    activity_log.append({
        "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Page": current_page,
        "Action": "Visited page",
        "Details": "Opened this section",
    })

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## CAMPUS IQ")
    st.caption("Smart classroom insights, made understandable.")
    st.divider()

    page = st.radio(
        "Where would you like to go?",
        PAGES,
        key="main_page",
    )

    st.divider()
    st.caption(
        "Prototype · Predictions are estimates, not confirmed live measurements."
    )


# ============================================================
# HOME
# ============================================================

if page == "Home":
    hero(
        "SMARTER CLASSROOM DECISIONS",
        "Understand your classrooms.",
        "Explore occupancy patterns, check an estimate, and review the timetable.",
    )

    if ALL_DATA.empty:
        st.warning(
            "No datasets were loaded. Check that your training and test files "
            "are saved inside the project's data folder."
        )
    else:
        total = len(ALL_DATA)
        occupied = (
            int((ALL_DATA["Occupancy"] == 1).sum())
            if "Occupancy" in ALL_DATA else 0
        )
        share = (
            occupied / total * 100
            if total and "Occupancy" in ALL_DATA else 0
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            card("RECORDS AVAILABLE", f"{total:,}", "Loaded dataset records")
        with c2:
            card("OCCUPIED LABELS", f"{occupied:,}", "Labels in the loaded records")
        with c3:
            card("OCCUPIED SHARE", f"{share:.1f}%", "Share of labelled records")

        st.markdown(
            '<div class="note">These statistics describe the supplied dataset, '
            'not the current occupancy of your college classrooms.</div>',
            unsafe_allow_html=True,
        )

        section("A quick look at the data")
        dataset_name = _persistent_selectbox(
            "Choose a dataset", list(DATASETS.keys()), key="home_dataset"
        )
        df = DATASETS[dataset_name]

        c1, c2 = st.columns(2)
        with c1:
            fig = occupancy_chart(df, "Occupancy labels")
            if fig:
                _show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("Occupancy labels are unavailable in this dataset.")

        with c2:
            fig = hourly_chart(df)
            if fig:
                _show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("A usable timestamp and occupancy label are needed for the hourly chart.")

        st.caption(
            "The first chart counts records in each class. The second shows "
            "the percentage of records labelled occupied at each hour."
        )

    section("Quick navigation")
    
    pages = [
        ("Check a Classroom", "Check a classroom"),
        ("Occupancy Trends", "Occupancy trends"),
        ("Visualization Explorer", "Explore visualizations"),
        ("Timetable & Editor", "Timetable & editor"),
        ("Reports & Downloads", "Reports & downloads"),
        ("Technical Insights", "Technical insights"),
        ("About CAMPUS IQ", "About CAMPUS IQ"),
    ]

    for i in range(0, len(pages), 2):
        cols = st.columns(2)

        for j, col in enumerate(cols):
            index = i + j

            if index < len(pages):
                page_name, button_label = pages[index]

                with col:
                    st.button(
                        button_label,
                        key=f"quick_nav_{index}",
                        use_container_width=True,
                        on_click=lambda p=page_name: st.session_state.update(
                            {"main_page": p}
                        ),
                    )


    with st.expander("About the datasets"):
        st.write(
            "The public occupancy dataset contains environmental readings and "
            "occupancy labels. It is used here to study patterns and train a model. "
            "It is not automatically connected to live campus rooms."
        )


# ============================================================
# CHECK A CLASSROOM
# ============================================================

elif page == "Check a Classroom":
    hero(
        "A PRACTICAL ESTIMATE",
        "Check a classroom",
        "Enter readings to estimate whether they resemble occupied conditions.",
    )

    if MODEL is None:
        st.error(
            "A saved model could not be loaded. Expected "
            "models/occupancy_model.pkl or "
            "models/timetable_occupancy_model.pkl."
        )
        if MODEL_SOURCE:
            st.caption(str(MODEL_SOURCE))
    else:
        st.caption(
            "This form uses manually entered values. It is not connected to live sensors."
        )

        room = _persistent_text_input("Classroom label", key="classroom_room", placeholder="e.g. A101")

        c1, c2 = st.columns(2)
        with c1:
            temperature = _persistent_number_input("Temperature (°C)", key="classroom_temperature", min_value=-10.0, max_value=60.0, value=23.0)
            humidity = _persistent_number_input("Humidity (%)", key="classroom_humidity", min_value=0.0, max_value=100.0, value=27.0)
            light = _persistent_number_input("Light reading", key="classroom_light", min_value=0.0, max_value=10000.0, value=450.0)

        with c2:
            co2 = _persistent_number_input("CO₂ reading (ppm)", key="classroom_co2", min_value=0.0, max_value=10000.0, value=600.0)
            humidity_ratio = _persistent_number_input("Humidity ratio", key="classroom_humidity_ratio", min_value=0.0, max_value=1.0, value=0.0045, format="%.5f")
            chosen_date = _persistent_date_input("Date", key="classroom_date", value=date.today())
            chosen_time = _persistent_time_input("Time", key="classroom_time", value=time(10, 30))

        submitted = st.button("Estimate occupancy", type="primary", use_container_width=True)

        if submitted:
            if not room.strip():
                st.error("Enter a classroom label first.")
            else:
                readings = {
                    "Temperature": temperature,
                    "Humidity": humidity,
                    "Light": light,
                    "CO2": co2,
                    "HumidityRatio": humidity_ratio,
                }
                when = datetime.combine(chosen_date, chosen_time)

                try:
                    predicted, confidence = predict_occupancy(
                        MODEL, readings, when
                    )
                    status = "Occupied" if predicted == 1 else "Not occupied"

                    result = {
                        "Room": room.strip(),
                        "Date and time": when.strftime("%Y-%m-%d %H:%M"),
                        **readings,
                        "Model used": "SVM",
                        "Estimated occupancy": status,
                        "Model score (%)": (
                            round(confidence, 2)
                            if confidence is not None else None
                        ),
                    }

                    st.session_state["last_estimate"] = result
                    st.session_state.setdefault("campus_iq_prediction_history", []).append(result.copy())

                    section("Your result")

                    if predicted == 1:
                        st.success(
                            "The model estimates that these readings resemble "
                            "occupied conditions."
                        )
                    else:
                        st.info(
                            "The model estimates that these readings resemble "
                            "unoccupied conditions."
                        )

                    if confidence is not None:
                        st.metric(
                            "Model score for this estimate",
                            f"{confidence:.1f}%",
                        )
                        st.caption(
                            "This score is not a guarantee of correctness or "
                            "a probability that a real classroom is occupied."
                        )

                    section("Timetable context")
                    schedule_row, message = check_schedule(
                        TIMETABLE, room, when
                    )

                    if schedule_row is not None:
                        subject = schedule_row.get("Subject", "Class")
                        st.write(
                            f"Matching entry: **{subject}**, "
                            f"**{schedule_row['Start_Time']}–"
                            f"{schedule_row['End_Time']}**."
                        )

                        scheduled_value = schedule_row.get(
                            "Class_Scheduled", True
                        )
                        scheduled = str(scheduled_value).strip().lower() not in {
                            "false", "no", "0", "not scheduled"
                        }

                        if scheduled and predicted == 0:
                            st.warning(
                                "A class is scheduled but the model estimates "
                                "unoccupied conditions. Verify manually."
                            )
                        elif not scheduled and predicted == 1:
                            st.warning(
                                "The model estimates occupied conditions when "
                                "no class is marked. Verify manually."
                            )
                        else:
                            st.info(
                                "The estimate and timetable do not show an "
                                "obvious mismatch. This is not live confirmation."
                            )
                    else:
                        st.info(message)

                    st.markdown(
                        '<div class="note">Do not switch off equipment based '
                        'on this estimate alone. This prototype does not measure '
                        'actual people, equipment status or electricity use.</div>',
                        unsafe_allow_html=True,
                    )

                    st.download_button(
                        "Download this estimate (CSV)",
                        pd.DataFrame([result]).to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="campus_iq_classroom_estimate.csv",
                        mime="text/csv",
                    )

                except Exception as exc:
                    st.error(f"Prediction could not be completed: {exc}")
                    st.info(
                        "The saved model's required input features must match "
                        "the fields in this form. Check the training pipeline."
                    )

    if st.session_state.get("last_estimate"):
        section("Latest estimate")
        _show_dataframe(
            pd.DataFrame([st.session_state["last_estimate"]]),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# OCCUPANCY TRENDS
# ============================================================

elif page == "Occupancy Trends":
    hero(
        "EXPLORE THE PATTERNS",
        "Occupancy trends",
        "Understand how the supplied data changes across records and time.",
    )

    if not DATASETS:
        st.warning("No datasets are available.")
    else:
        name = _persistent_selectbox("Dataset", list(DATASETS.keys()), key="trends_dataset")
        df = DATASETS[name]

        c1, c2 = st.columns(2)

        with c1:
            fig = occupancy_chart(df)
            if fig:
                _show_plotly_chart(fig, use_container_width=True)

        with c2:
            fig = hourly_chart(df)
            if fig:
                _show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("Hourly analysis is unavailable for this dataset.")

        features = [
            c for c in [
                "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"
            ] if c in df.columns
        ]

        if features:
            feature = _persistent_selectbox("Explore a sensor", features, key="trends_sensor")
            fig = sensor_chart(df, feature)
            if fig:
                _show_plotly_chart(fig, use_container_width=True)
                st.caption(
                    "This chart shows recorded sensor changes. It does not "
                    "establish cause and effect."
                )

        st.download_button(
            "Download selected dataset (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )


# ============================================================
# VISUALIZATION EXPLORER
# ============================================================

elif page == "Visualization Explorer":
    hero(
        "INTERACTIVE DATA DISCOVERY",
        "Explore every angle of the data",
        "Choose a dataset, decide what you want to understand, and switch chart styles.",
    )

    if not DATASETS:
        st.error("No datasets were loaded. Check the files inside the data folder.")
    else:
        dataset_name = _persistent_selectbox("1. Choose a dataset", list(DATASETS.keys()), key="viz_dataset")
        original = DATASETS[dataset_name].copy()

        sensor_features = [
            c for c in [
                "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"
            ] if c in original.columns
        ]
        numeric_features = [
            c for c in original.select_dtypes(include="number").columns
            if c not in ["hour", "day_of_week"]
        ]

        section("2. What do you want to understand?")

        questions = [
            "Occupancy overview",
            "Sensor behaviour over time",
            "Compare sensor distributions",
            "Compare occupied vs unoccupied",
            "Relationship between two sensors",
            "Correlation between sensors",
            "Occupancy by hour and day",
            "Explore the actual records",
        ]

        question = _persistent_selectbox("Choose a question", questions, key="viz_question")

        chart_options = {
            "Occupancy overview": ["Bar chart", "Donut chart"],
            "Sensor behaviour over time": ["Line chart", "Area chart"],
            "Compare sensor distributions": ["Histogram", "Box plot"],
            "Compare occupied vs unoccupied": ["Box plot", "Violin plot"],
            "Relationship between two sensors": ["Scatter plot"],
            "Correlation between sensors": ["Heatmap"],
            "Occupancy by hour and day": ["Heatmap", "Grouped bar chart"],
            "Explore the actual records": ["Data table"],
        }

        chart_type = _persistent_selectbox("3. Choose a visualization style", chart_options[question], key="viz_chart_type")

        filtered = original.copy()

        if "Occupancy" in filtered.columns:
            labels = _persistent_multiselect(
                "Optional: filter occupancy labels", options=[0, 1], default=[0, 1],
                format_func=lambda x: ("Not occupied" if x == 0 else "Occupied"), key="viz_labels"
            )
            filtered = filtered[filtered["Occupancy"].isin(labels)]

        if filtered.empty:
            st.warning(
                "No records match the current filters. Select an occupancy "
                "label or change the dataset."
            )
        else:
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Records displayed", f"{len(filtered):,}")
            with c2:
                st.metric("Sensor features", len(sensor_features))
            with c3:
                occupied_n = (
                    int((filtered["Occupancy"] == 1).sum())
                    if "Occupancy" in filtered else None
                )
                st.metric(
                    "Occupied labels",
                    f"{occupied_n:,}" if occupied_n is not None else "Unavailable",
                )

            fig = None

            # Occupancy overview
            if question == "Occupancy overview":
                counts = occupancy_counts(filtered)

                if counts.empty:
                    st.info("Occupancy labels are not available.")
                elif chart_type == "Donut chart":
                    fig = px.pie(
                        counts,
                        names="Occupancy",
                        values="Records",
                        hole=0.55,
                        title="How the records are split by occupancy",
                        color="Occupancy",
                        color_discrete_map={
                            "Not occupied": "#087E8B",
                            "Occupied": "#F4A261",
                        },
                    )
                else:
                    fig = px.bar(
                        counts,
                        x="Occupancy",
                        y="Records",
                        text="Records",
                        title="Number of records in each occupancy class",
                        color="Occupancy",
                        color_discrete_map={
                            "Not occupied": "#087E8B",
                            "Occupied": "#F4A261",
                        },
                    )
                    fig.update_traces(textposition="outside")

            # Sensor behaviour over time
            elif question == "Sensor behaviour over time":
                if "date" not in filtered.columns:
                    st.info("Usable timestamps are unavailable.")
                elif not sensor_features:
                    st.info("No supported sensor columns are available.")
                else:
                    feature = _persistent_selectbox("Choose a sensor", sensor_features, key="viz_time_feature")

                    plot_df = filtered.dropna(
                        subset=["date", feature]
                    ).sort_values("date").copy()

                    if len(plot_df) > 3000:
                        step = max(1, len(plot_df) // 3000)
                        plot_df = plot_df.iloc[::step]

                    if plot_df.empty:
                        st.info("There are no usable values to plot.")
                    elif chart_type == "Area chart":
                        fig = px.area(
                            plot_df,
                            x="date",
                            y=feature,
                            title=f"{feature}: change over recorded time",
                        )
                    else:
                        fig = px.line(
                            plot_df,
                            x="date",
                            y=feature,
                            title=f"{feature}: change over recorded time",
                        )

            # Sensor distributions
            elif question == "Compare sensor distributions":
                if not sensor_features:
                    st.info("No supported sensor columns are available.")
                else:
                    feature = _persistent_selectbox("Choose a sensor", sensor_features, key="viz_distribution_feature")

                    if chart_type == "Histogram":
                        if "Occupancy" in filtered.columns:
                            plot_df = filtered.copy()
                            plot_df["Occupancy group"] = plot_df[
                                "Occupancy"
                            ].map({
                                0: "Not occupied",
                                1: "Occupied",
                            })
                            fig = px.histogram(
                                plot_df,
                                x=feature,
                                color="Occupancy group",
                                barmode="overlay",
                                opacity=0.7,
                                nbins=35,
                                title=f"Distribution of {feature} readings",
                                color_discrete_map={
                                    "Not occupied": "#087E8B",
                                    "Occupied": "#F4A261",
                                },
                            )
                        else:
                            fig = px.histogram(
                                filtered,
                                x=feature,
                                nbins=35,
                                title=f"Distribution of {feature} readings",
                            )
                    else:
                        fig = px.box(
                            filtered,
                            y=feature,
                            points=False,
                            title=f"Spread of {feature} readings",
                        )

            # Occupied vs unoccupied
            elif question == "Compare occupied vs unoccupied":
                if "Occupancy" not in filtered.columns:
                    st.info("Occupancy labels are required for this comparison.")
                elif not sensor_features:
                    st.info("No supported sensor columns are available.")
                else:
                    feature = _persistent_selectbox("Choose a sensor", sensor_features, key="viz_compare_feature")

                    plot_df = filtered.copy()
                    plot_df["Occupancy group"] = plot_df[
                        "Occupancy"
                    ].map({0: "Not occupied", 1: "Occupied"})

                    if chart_type == "Violin plot":
                        fig = px.violin(
                            plot_df,
                            x="Occupancy group",
                            y=feature,
                            color="Occupancy group",
                            box=True,
                            title=f"{feature}: compare occupancy groups",
                            color_discrete_map={
                                "Not occupied": "#087E8B",
                                "Occupied": "#F4A261",
                            },
                        )
                    else:
                        fig = px.box(
                            plot_df,
                            x="Occupancy group",
                            y=feature,
                            color="Occupancy group",
                            title=f"{feature}: compare occupancy groups",
                            color_discrete_map={
                                "Not occupied": "#087E8B",
                                "Occupied": "#F4A261",
                            },
                        )

            # Relationship between two sensors
            elif question == "Relationship between two sensors":
                if len(sensor_features) < 2:
                    st.info("At least two sensor features are needed.")
                else:
                    x_feature = _persistent_selectbox("Horizontal axis", sensor_features, key="viz_scatter_x")
                    y_options = [
                        f for f in sensor_features if f != x_feature
                    ]
                    y_feature = _persistent_selectbox("Vertical axis", y_options, key="viz_scatter_y")

                    plot_df = filtered
                    if len(plot_df) > 5000:
                        plot_df = plot_df.sample(
                            5000, random_state=42
                        )

                    if "Occupancy" in plot_df.columns:
                        plot_df = plot_df.copy()
                        plot_df["Occupancy group"] = plot_df[
                            "Occupancy"
                        ].map({
                            0: "Not occupied",
                            1: "Occupied",
                        })
                        fig = px.scatter(
                            plot_df,
                            x=x_feature,
                            y=y_feature,
                            color="Occupancy group",
                            opacity=0.65,
                            title=f"{x_feature} compared with {y_feature}",
                            color_discrete_map={
                                "Not occupied": "#087E8B",
                                "Occupied": "#F4A261",
                            },
                        )
                    else:
                        fig = px.scatter(
                            plot_df,
                            x=x_feature,
                            y=y_feature,
                            opacity=0.65,
                            title=f"{x_feature} compared with {y_feature}",
                        )

            # Correlation heatmap
            elif question == "Correlation between sensors":
                if len(numeric_features) < 2:
                    st.info("At least two numeric features are needed.")
                else:
                    corr = filtered[numeric_features].corr()
                    fig = px.imshow(
                        corr,
                        text_auto=".2f",
                        zmin=-1,
                        zmax=1,
                        color_continuous_scale="RdBu_r",
                        title="Which numeric features move together?",
                        aspect="auto",
                    )

            # Day/hour
            elif question == "Occupancy by hour and day":
                if "date" not in filtered.columns:
                    st.info("Usable timestamps are unavailable.")
                elif "Occupancy" not in filtered.columns:
                    st.info("Occupancy labels are required for this view.")
                else:
                    plot_df = filtered.dropna(subset=["date"]).copy()
                    plot_df["Hour"] = plot_df["date"].dt.hour
                    plot_df["Day"] = plot_df["date"].dt.day_name()

                    day_order = [
                        "Monday", "Tuesday", "Wednesday", "Thursday",
                        "Friday", "Saturday", "Sunday",
                    ]

                    if chart_type == "Heatmap":
                        pivot = plot_df.pivot_table(
                            index="Day",
                            columns="Hour",
                            values="Occupancy",
                            aggfunc="mean",
                        ).reindex(day_order)

                        fig = px.imshow(
                            pivot * 100,
                            labels={
                                "x": "Hour of day",
                                "y": "Day",
                                "color": "Occupied records (%)",
                            },
                            zmin=0,
                            zmax=100,
                            color_continuous_scale="YlGnBu",
                            title="Occupied-label share by day and hour",
                            aspect="auto",
                        )
                    else:
                        hourly = plot_df.groupby(
                            ["Hour", "Day"], as_index=False
                        )["Occupancy"].mean()
                        hourly["Occupied records (%)"] = (
                            hourly["Occupancy"] * 100
                        )
                        fig = px.bar(
                            hourly,
                            x="Hour",
                            y="Occupied records (%)",
                            color="Day",
                            barmode="group",
                            title="Occupied-label share by hour and day",
                        )

            # Raw records
            elif question == "Explore the actual records":
                _show_dataframe(
                    filtered,
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download filtered records (CSV)",
                    filtered.to_csv(index=False).encode("utf-8"),
                    file_name="campus_iq_filtered_records.csv",
                    mime="text/csv",
                )
            # Save the current visualization settings for the PDF
            
                
            
            # Save the selected visualization for the PDF
            if fig is not None:
                fig.update_layout(**PLOT_LAYOUT)

                report_visualization = {
                    "dataset": dataset_name,
                    "question": question,
                    "chart_type": chart_type,
                    "filters": (
                        list(labels)
                        if "Occupancy" in filtered.columns
                        else []
                    ),
                    "records_displayed": len(filtered),
                    # Keep the actual selected Plotly figure as a frozen
                    # serializable snapshot; image export may be unavailable
                    # when the optional Kaleido renderer is not installed.
                    "plotly_json": fig.to_json(),
                    "explanation": "This graph reflects the dataset, question, chart type, and filters selected in Visualization Explorer.",
                }

                # Capture the selected chart for the PDF
                try:
                    report_visualization["image"] = fig.to_image(
                        format="png",
                        width=1000,
                        height=600,
                    )
                except Exception:
                    report_visualization["image"] = None

                # Store a snapshot of the exact chart and settings currently
                # shown to the user. Keep this separate from the explanatory
                # summary below so it cannot be overwritten by summary fields.
                st.session_state["report_visualization"] = dict(report_visualization)

                # Display the chart only once. The report-artifact helper also
                # captures its image/plot data for the generated PDF.
                _show_plotly_chart(
                    fig,
                    use_container_width=True,
                )

            explanations = {
                "Occupancy overview":
                    "Counts the records labelled occupied and not occupied. "
                    "It does not show how many people are in a real classroom now.",

                "Sensor behaviour over time":
                    "Shows how a sensor reading changes across recorded timestamps. "
                    "A pattern alone cannot explain why a change occurred.",

                "Compare sensor distributions":
                    "Shows the spread and frequency of sensor readings. A wider "
                    "spread means the values vary more in the selected records.",

                "Compare occupied vs unoccupied":
                    "Compares sensor readings between records with different "
                    "occupancy labels. It can reveal associations, not causation.",

                "Relationship between two sensors":
                    "Each point represents a record. A visible relationship means "
                    "the readings may move together; it does not prove causation.",

                "Correlation between sensors":
                    "Correlation ranges from -1 to +1. Positive values indicate a "
                    "positive linear relationship; negative values indicate an "
                    "inverse relationship; values near zero indicate little linear relationship.",

                "Occupancy by hour and day":
                    "Shows the percentage of records labelled occupied by recorded "
                    "day and hour. It is not a verified college timetable.",

                "Explore the actual records":
                    "Displays the selected underlying records for inspection and export.",
            }

            section("What this graph means")
            st.info(explanations[question])
            viz_summary = {
                "Dataset": dataset_name,
                "Question explored": question,
                "Chart type": chart_type,
                "Occupancy filters": labels if "Occupancy" in original.columns else "Not applicable",
                "Records displayed": int(len(filtered)),
                "Sensor over time": st.session_state.get("viz_time_feature", "Not applicable"),
                "Sensor distribution": st.session_state.get("viz_distribution_feature", "Not applicable"),
                "Sensor comparison": st.session_state.get("viz_compare_feature", "Not applicable"),
                "Horizontal axis": st.session_state.get("viz_scatter_x", "Not applicable"),
                "Vertical axis": st.session_state.get("viz_scatter_y", "Not applicable"),
                "Explanation": explanations[question],
            }
            # Merge the human-readable selection summary into the saved chart
            # snapshot. In particular, preserve the captured PNG and chart data;
            # replacing the snapshot with viz_summary here used to discard the
            # selected graph before the PDF was generated.
            _saved_visualization = dict(
                st.session_state.get("report_visualization", {}) or {}
            )
            _saved_visualization.update(viz_summary)
            _saved_visualization["explanation"] = explanations[question]
            st.session_state["report_visualization"] = _saved_visualization
            _record_report_artifact("Visualization Explorer", {
                "kind": "summary", "title": "Selected visualization and interpretation", "identity": "viz_summary",
                "data": viz_summary, "explanation": explanations[question],
            })

            st.caption(
                "Source: selected public dataset. Graphs do not represent live "
                "campus measurements or verified electricity use."
            )


# ============================================================
# TIMETABLE & EDITOR
# ============================================================

elif page == "Timetable & Editor":
    hero(
        "SCHEDULE MANAGEMENT",
        "Timetable & editor",
        "View, add, edit, remove and save timetable entries.",
    )

    st.caption(f"Current source: {TIMETABLE_SOURCE}")
    st.warning(
        "If you see SAMPLE entries, replace them with an authorised college "
        "timetable before relying on schedule comparisons."
    )

    if "timetable_editor_data" not in st.session_state:
        st.session_state["timetable_editor_data"] = TIMETABLE.copy()

    # Use a fresh widget key that is not pre-populated with a stale editor
    # state. Streamlit's editor state can have an older column layout; manually
    # assigning that state causes pandas' "length must be ..." error.
    _editor_key = "timetable_editor_widget_v2"
    edited = st.data_editor(
        st.session_state["timetable_editor_data"],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=_editor_key,
        column_config={
            "Class_Scheduled": st.column_config.CheckboxColumn(
                "Class scheduled?",
                default=True,
            ),
        },
    )

    st.session_state["timetable_editor_data"] = edited.copy()

    c1, c2 = st.columns(2)

    with c1:
        if st.button(
            "Save timetable changes",
            type="primary",
            use_container_width=True,
        ):
            try:
                save_timetable(edited)
                st.session_state["timetable_editor_data"] = edited.copy()
                st.cache_data.clear()
                st.success(f"Timetable saved to {TIMETABLE_PATH}")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save timetable: {exc}")

    with c2:
        st.download_button(
            "Download timetable (CSV)",
            edited.to_csv(index=False).encode("utf-8"),
            file_name="campus_iq_timetable.csv",
            mime="text/csv",
            use_container_width=True,
        )

    section("Check a room and time")

    required = {"Room", "Day", "Start_Time", "End_Time"}

    if required.issubset(edited.columns):
        rooms = sorted(
            edited["Room"].dropna().astype(str).unique().tolist()
        )

        if rooms:
            room = _persistent_selectbox("Room", rooms, key="schedule_room")
            day_name = _persistent_selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], key="schedule_day")
            check_time = _persistent_time_input("Time", key="schedule_check_time", value=time(10, 30))

            days = [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ]
            today = date.today()
            offset = (days.index(day_name) - today.weekday()) % 7
            selected_date = date.fromordinal(today.toordinal() + offset)

            row, message = check_schedule(
                edited,
                room,
                datetime.combine(selected_date, check_time),
            )
            st.session_state["last_schedule_check"] = {
                "Room": room,
                "Day": day_name,
                "Time": check_time.isoformat(),
                "Result": (f"Scheduled entry: {row.get('Subject', 'Class')} ({row['Start_Time']}–{row['End_Time']})" if row is not None else message),
            }
            _record_report_artifact("Timetable & Editor", {
                "kind": "summary", "title": "Latest timetable check", "identity": "latest_timetable_check",
                "data": st.session_state["last_schedule_check"],
                "explanation": "Result generated for the selected room, day, and time.",
            })

            if row is not None:
                st.success(
                    f"Scheduled entry: {row.get('Subject', 'Class')} "
                    f"({row['Start_Time']}–{row['End_Time']})."
                )
            else:
                st.info(message)
    else:
        st.error(
            "Required columns: Room, Day, Start_Time and End_Time."
        )


# ============================================================
# REPORTS & DOWNLOADS
# ============================================================

elif page == "Reports & Downloads":
    hero(
        "READABLE RESULTS",
        "Reports & downloads",
        "A human-readable report, supporting tables and downloadable data.",
    )

    st.write(
        "The PDF explains the findings in plain English and includes charts "
        "and available analysis tables. Excel contains the available supporting "
        "data. CSV is useful when you need the raw records."
    )

    dataset_summary = pd.DataFrame([
        {
            "Dataset": name,
            "Records": len(df),
            "Occupied labels": (
                int((df["Occupancy"] == 1).sum())
                if "Occupancy" in df else None
            ),
            "Not occupied labels": (
                int((df["Occupancy"] == 0).sum())
                if "Occupancy" in df else None
            ),
        }
        for name, df in DATASETS.items()
    ])

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "Download data (CSV)",
            ALL_DATA.to_csv(index=False).encode("utf-8")
            if not ALL_DATA.empty else b"",
            file_name="CAMPUS_IQ_Data.csv",
            mime="text/csv",
            disabled=ALL_DATA.empty,
            use_container_width=True,
        )

    with c2:
        try:
            pdf_bytes = build_pdf(
                DATASETS,
                RESULTS,
                QUALITY,
                FEATURES,
                st.session_state.get("timetable_editor_data", TIMETABLE),
                collect_user_report_selections(),
                st.session_state.get("last_estimate"),
                st.session_state.get("report_visualization"),
            )

            st.download_button(
                "Download readable report (PDF)",
                pdf_bytes,
                file_name="CAMPUS_IQ_Report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"PDF report generation failed: {exc}")

    with c3:
        try:
            excel_bytes = build_excel({
                "Dataset summary": dataset_summary,
                "Model comparison": RESULTS,
                "Evaluation metrics": DIAGNOSTICS,
                "ROC AUC": ROC_RESULTS,
                "Feature importance": FEATURES,
                "Data quality": QUALITY,
                "Sensor statistics": SENSOR_SUMMARY,
                "Timetable": TIMETABLE,
            })
            st.download_button(
                "Download analysis (Excel)",
                excel_bytes,
                file_name="CAMPUS_IQ_Analysis.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Excel report generation failed: {exc}")

    if not dataset_summary.empty:
        section("Dataset summary")
        _show_dataframe(
            dataset_summary,
            use_container_width=True,
            hide_index=True,
        )

    if not RESULTS.empty:
        section("Saved model evaluation")
        _show_dataframe(
            RESULTS,
            use_container_width=True,
            hide_index=True,
        )

    if st.session_state.get("last_estimate"):
        section("Latest classroom estimate")
        _show_dataframe(
            pd.DataFrame([st.session_state["last_estimate"]]),
            use_container_width=True,
            hide_index=True,
        )

    if not ALL_DATA.empty:
        st.download_button(
            "Download all loaded records (CSV)",
            ALL_DATA.to_csv(index=False).encode("utf-8"),
            file_name="CAMPUS_IQ_All_Records.csv",
            mime="text/csv",
        )


# ============================================================
# TECHNICAL INSIGHTS
# ============================================================

elif page == "Technical Insights":
    hero(
        "OPTIONAL TECHNICAL WORKSPACE",
        "Technical insights",
        "Evaluation evidence for faculty, developers and project reviewers.",
    )

    if RESULTS.empty:
        st.info(
            "No saved model comparison table was found. Run the project's "
            "model evaluation scripts to create it."
        )
    else:
        section("Saved model comparison")
        _show_dataframe(
            RESULTS,
            use_container_width=True,
            hide_index=True,
        )

        numeric_cols = [
            c for c in RESULTS.columns
            if pd.api.types.is_numeric_dtype(RESULTS[c])
        ]

        if numeric_cols and len(RESULTS.columns) > 1:
            metric = _persistent_selectbox("Choose a metric to compare", numeric_cols, key="technical_metric")
            label_col = RESULTS.columns[0]
            plot_df = RESULTS[[label_col, metric]].copy()
            plot_df[metric] = pd.to_numeric(
                plot_df[metric], errors="coerce"
            )
            plot_df = plot_df.dropna()

            if not plot_df.empty:
                fig = px.bar(
                    plot_df,
                    x=label_col,
                    y=metric,
                    text=metric,
                    title=f"Model comparison: {metric}",
                )
                fig.update_layout(**PLOT_LAYOUT)
                fig.update_xaxes(tickangle=-30)
                _show_plotly_chart(fig, use_container_width=True)

    technical_tables = [
        ("Evaluation metrics", DIAGNOSTICS),
        ("ROC-AUC results", ROC_RESULTS),
        ("Feature importance", FEATURES),
        ("Data quality report", QUALITY),
        ("Sensor summary statistics", SENSOR_SUMMARY),
    ]

    for title, df in technical_tables:
        section(title)
        if df.empty:
            st.info(f"No saved results available for {title}.")
        else:
            _show_dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("What do the evaluation metrics mean?"):
        st.markdown("""
        - **Accuracy:** proportion of all predictions that were correct.
        - **Precision:** among predicted occupied cases, how many were correct.
        - **Recall:** among actually occupied cases, how many were identified.
        - **F1-score:** balances precision and recall.
        - **ROC-AUC:** measures how well a model separates classes across thresholds.
        - **Feature importance:** indicates which inputs matter under the saved
          analysis method. It does not prove cause and effect.
        """)

    if MODEL_SOURCE:
        st.caption(f"Loaded model source: {MODEL_SOURCE}")


# ============================================================
# ABOUT
# ============================================================

elif page == "About CAMPUS IQ":
    hero(
        "ABOUT THE PROJECT",
        "Useful decisions, grounded in evidence.",
        "A machine-learning prototype for occupancy estimates and schedule-aware review.",
    )

    section("For everyday users")
    st.write(
        "Check an estimate, explore understandable trends, manage a timetable, "
        "and download a readable report. Technical details are available separately."
    )

    section("What the prototype can do")
    st.markdown("""
    - Estimate occupied or unoccupied labels from supported sensor inputs.
    - Explore patterns in the supplied dataset.
    - Compare an estimate with timetable entries for manual review.
    - Allow timetable entries to be edited and saved.
    - Export available records and reports.
    """)

    section("What it cannot confirm")
    st.markdown("""
    - Live occupancy without an authorised live sensor connection.
    - Whether lights or air conditioning are actually switched on.
    - Actual electricity consumption or verified energy savings.
    - Whether a sample timetable represents the official campus schedule.
    """)

    section("Future improvements")
    st.write(
        "Authorised live sensor integration, verified timetable integration, "
        "equipment-status measurements, electricity-meter data and validation "
        "using real campus observations."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "CAMPUS IQ · Smart Classroom Occupancy & Energy Advisor · "
    "Prototype decision support, not live building control"
)
