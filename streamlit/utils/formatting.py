"""
Formatting and Styling Helpers for Streamlit Insurance Dashboard.
"""

def format_currency(val: float) -> str:
    """Formats numeric value as US Dollar currency."""
    if val is None:
        return "$0.00"
    return f"${val:,.2f}" if abs(val) < 100000 else f"${val:,.0f}"


def format_percentage(val: float) -> str:
    """Formats ratio or percentage."""
    if val is None:
        return "0.0%"
    return f"{val:.1f}%"


def format_number(val: int) -> str:
    """Formats integer with commas."""
    if val is None:
        return "0"
    return f"{int(val):,}"


def get_risk_badge_html(risk_level: str) -> str:
    """Returns a styled HTML badge for the risk level."""
    color_map = {
        "LOW RISK": {
            "bg": "#d4edda",
            "text": "#155724",
            "border": "#c3e6cb",
            "icon": "✅",
        },
        "MEDIUM RISK": {
            "bg": "#fff3cd",
            "text": "#856404",
            "border": "#ffeeba",
            "icon": "⚠️",
        },
        "HIGH RISK": {
            "bg": "#f8d7da",
            "text": "#721c24",
            "border": "#f5c6cb",
            "icon": "🚨",
        },
    }
    style = color_map.get(risk_level, color_map["LOW RISK"])
    return f"""
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 8px;
        background-color: {style['bg']};
        color: {style['text']};
        border: 1px solid {style['border']};
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    ">
        <span>{style['icon']}</span>
        <span>{risk_level}</span>
    </div>
    """
