"""
Custom Styles for Smart PDF Parser Streamlit Application

Provides custom CSS styling and visual enhancements for the application.
"""

import streamlit as st
from typing import Optional

def apply_custom_styles():
    """Apply custom CSS styles to the Streamlit application."""
    
    custom_css = """
    <style>
    /* Main application styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main .block-container {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
        padding: 2rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .sidebar .sidebar-content .stSelectbox label {
        color: white !important;
    }
    
    .sidebar .sidebar-content .stButton > button {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .sidebar .sidebar-content .stButton > button:hover {
        background-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Header styling */
    h1 {
        color: #2E86AB;
        text-align: center;
        font-weight: 700;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2 {
        color: #A23B72;
        border-bottom: 2px solid #A23B72;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    h3 {
        color: #F18F01;
        margin-bottom: 1rem;
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
    }
    
    .metric-container .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .metric-container .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(45deg, #f093fb, #f5576c);
    }
    
    .stButton > button[kind="secondary"] {
        background: linear-gradient(45deg, #a8edea, #fed6e3);
        color: #333;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 5px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
    }
    
    .stAlert[data-baseweb="notification"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .dataframe th {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
    }
    
    /* Element type badges */
    .element-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .element-badge.text {
        background-color: #2E86AB;
        color: white;
    }
    
    .element-badge.heading {
        background-color: #A23B72;
        color: white;
    }
    
    .element-badge.table {
        background-color: #F18F01;
        color: white;
    }
    
    .element-badge.list {
        background-color: #C73E1D;
        color: white;
    }
    
    .element-badge.image {
        background-color: #0F7B0F;
        color: white;
    }
    
    .element-badge.caption {
        background-color: #7209B7;
        color: white;
    }
    
    .element-badge.formula {
        background-color: #F72585;
        color: white;
    }
    
    .element-badge.code {
        background-color: #4361EE;
        color: white;
    }
    
    /* Confidence indicators */
    .confidence-indicator {
        display: inline-block;
        width: 100px;
        height: 20px;
        border-radius: 10px;
        position: relative;
        background-color: #e0e0e0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .confidence-high {
        background: linear-gradient(90deg, #4CAF50, #45a049);
    }
    
    .confidence-medium {
        background: linear-gradient(90deg, #FF9800, #f57c00);
    }
    
    .confidence-low {
        background: linear-gradient(90deg, #f44336, #d32f2f);
    }
    
    /* Verification status indicators */
    .verification-status {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .verification-status.pending {
        background-color: #FFF3E0;
        color: #FF6F00;
        border: 1px solid #FFB74D;
    }
    
    .verification-status.correct {
        background-color: #E8F5E8;
        color: #2E7D32;
        border: 1px solid #81C784;
    }
    
    .verification-status.incorrect {
        background-color: #FFEBEE;
        color: #C62828;
        border: 1px solid #E57373;
    }
    
    .verification-status.partial {
        background-color: #FFF8E1;
        color: #F57F17;
        border: 1px solid #FFCC02;
    }
    
    /* Search result highlighting */
    .search-highlight {
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 600;
    }
    
    /* Animation classes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-in-out;
    }
    
    /* Loading spinner */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 2s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Page transition effects */
    .page-transition {
        animation: fadeIn 0.3s ease-in-out;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        .metric-container {
            margin-bottom: 1rem;
        }
        
        .stButton > button {
            width: 100%;
            margin-bottom: 0.5rem;
        }
    }
    
    /* Print styles */
    @media print {
        .sidebar {
            display: none !important;
        }
        
        .main .block-container {
            box-shadow: none;
            background: white;
        }
    }
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)


def create_element_badge(element_type: str, confidence: Optional[float] = None) -> str:
    """Create a styled badge for element type.
    
    Args:
        element_type: Type of the element
        confidence: Confidence score (optional)
        
    Returns:
        HTML string for the badge
    """
    badge_html = f'<span class="element-badge {element_type}">{element_type.title()}</span>'
    
    if confidence is not None:
        conf_class = "high" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "low"
        badge_html += f'''
        <div class="confidence-indicator" title="Confidence: {confidence:.3f}">
            <div class="confidence-fill confidence-{conf_class}" 
                 style="width: {confidence * 100}%"></div>
        </div>
        '''
    
    return badge_html


def create_verification_badge(status: str) -> str:
    """Create a styled badge for verification status.
    
    Args:
        status: Verification status
        
    Returns:
        HTML string for the badge
    """
    status_icons = {
        'pending': '⏳',
        'correct': '✅',
        'incorrect': '❌',
        'partial': '⚠️'
    }
    
    icon = status_icons.get(status, '❓')
    
    return f'''
    <span class="verification-status {status}">
        {icon} {status.title()}
    </span>
    '''


def create_progress_indicator(progress: float, label: str = "") -> str:
    """Create a styled progress indicator.
    
    Args:
        progress: Progress value (0-1)
        label: Optional label
        
    Returns:
        HTML string for the progress indicator
    """
    percentage = int(progress * 100)
    color = "high" if progress >= 0.8 else "medium" if progress >= 0.5 else "low"
    
    return f'''
    <div style="text-align: center; margin: 1rem 0;">
        {f"<div><strong>{label}</strong></div>" if label else ""}
        <div class="confidence-indicator" style="width: 200px; margin: 0.5rem auto;">
            <div class="confidence-fill confidence-{color}" 
                 style="width: {percentage}%"></div>
        </div>
        <div style="font-size: 0.9rem; color: #666;">{percentage}%</div>
    </div>
    '''


def add_page_transition():
    """Add page transition animation."""
    st.markdown(
        '<div class="page-transition">',
        unsafe_allow_html=True
    )


def close_page_transition():
    """Close page transition div."""
    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )