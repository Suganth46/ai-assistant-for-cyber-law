from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import random

load_dotenv()

# Initialize MongoDB connection
client = MongoClient(os.getenv('MONGODB_URI'))
db = client['legal_analytics']

def init_sample_data():
    """Initialize sample data for the analytics dashboard"""
    if db.cases.count_documents({}) == 0:
        # Create sample data
        categories = ['contracts', 'litigation', 'regulatory']
        sample_data = []
        
        # Generate data for the last 30 days
        for i in range(30):
            date = datetime.now() - timedelta(days=i)
            # Generate 5-15 cases per day
            num_cases = random.randint(5, 15)
            
            for _ in range(num_cases):
                category = random.choice(categories)
                success_rate = random.uniform(0.5, 1.0)
                processing_time = random.uniform(1, 30)
                
                case = {
                    'created_at': date,
                    'category': category,
                    'success_rate': success_rate,
                    'processing_time': processing_time,
                    'status': random.choice(['pending', 'completed', 'in_progress']),
                    'priority': random.choice(['low', 'medium', 'high']),
                    'client_type': random.choice(['individual', 'corporate', 'government']),
                    'region': random.choice(['north', 'south', 'east', 'west', 'central'])
                }
                sample_data.append(case)
        
        # Insert sample data
        if sample_data:
            db.cases.insert_many(sample_data)
            print(f"Inserted {len(sample_data)} sample cases")

# Initialize sample data
init_sample_data()

def create_dashboard(server):
    """Create and configure the Dash application"""
    app = Dash(__name__, server=server, url_base_pathname='/analytics/')
    
    # Layout
    app.layout = html.Div([
        html.H1('Legal Analytics Dashboard', className='dashboard-title'),
        
        # Filters
        html.Div([
            html.Div([
                html.Label('Date Range'),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                )
            ], className='filter-item'),
            
            html.Div([
                html.Label('Category'),
                dcc.Dropdown(
                    id='category-filter',
                    options=[
                        {'label': 'All Categories', 'value': 'all'},
                        {'label': 'Contracts', 'value': 'contracts'},
                        {'label': 'Litigation', 'value': 'litigation'},
                        {'label': 'Regulatory', 'value': 'regulatory'}
                    ],
                    value='all'
                )
            ], className='filter-item')
        ], className='filters-container'),
        
        # Main content
        html.Div([
            # Top row
            html.Div([
                html.Div([
                    html.H3('Case Volume Trends'),
                    dcc.Graph(id='case-volume-chart')
                ], className='chart-container'),
                
                html.Div([
                    html.H3('Category Distribution'),
                    dcc.Graph(id='category-pie-chart')
                ], className='chart-container')
            ], className='row'),
            
            # Bottom row
            html.Div([
                html.Div([
                    html.H3('Success Rate Analysis'),
                    dcc.Graph(id='success-rate-chart')
                ], className='chart-container'),
                
                html.Div([
                    html.H3('Processing Time Trends'),
                    dcc.Graph(id='processing-time-chart')
                ], className='chart-container')
            ], className='row')
        ], className='charts-container')
    ])
    
    # Callbacks
    @app.callback(
        [Output('case-volume-chart', 'figure'),
         Output('category-pie-chart', 'figure'),
         Output('success-rate-chart', 'figure'),
         Output('processing-time-chart', 'figure')],
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date'),
         Input('category-filter', 'value')]
    )
    def update_charts(start_date, end_date, category):
        try:
            # Fetch data from MongoDB
            query = {}
            if start_date and end_date:
                query['created_at'] = {
                    '$gte': datetime.fromisoformat(start_date),
                    '$lte': datetime.fromisoformat(end_date)
                }
            if category != 'all':
                query['category'] = category
                
            data = list(db.cases.find(query))
            
            if not data:
                # Return empty figures if no data
                empty_fig = go.Figure()
                empty_fig.add_annotation(
                    text="No data available for the selected filters",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return empty_fig, empty_fig, empty_fig, empty_fig
            
            df = pd.DataFrame(data)
            
            # Ensure date column exists and is in datetime format
            if 'created_at' in df.columns:
                df['date'] = pd.to_datetime(df['created_at']).dt.date
            else:
                df['date'] = pd.to_datetime(df['_id'].astype(str).str[:8], format='%Y%m%d').dt.date
            
            # Case Volume Trends
            volume_data = df.groupby('date').size().reset_index(name='count')
            volume_fig = px.line(
                volume_data,
                x='date',
                y='count',
                title='Case Volume Over Time'
            )
            volume_fig.update_layout(xaxis_title='Date', yaxis_title='Number of Cases')
            
            # Category Distribution
            if 'category' in df.columns:
                category_counts = df['category'].value_counts().reset_index()
                category_counts.columns = ['category', 'count']
                category_fig = px.pie(
                    category_counts,
                    names='category',
                    values='count',
                    title='Case Distribution by Category'
                )
            else:
                category_fig = go.Figure()
                category_fig.add_annotation(
                    text="Category data not available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            # Success Rate Analysis
            if 'success_rate' in df.columns:
                success_data = df.groupby('category')['success_rate'].mean().reset_index()
                success_fig = px.bar(
                    success_data,
                    x='category',
                    y='success_rate',
                    title='Success Rate by Category'
                )
                success_fig.update_layout(xaxis_title='Category', yaxis_title='Success Rate')
            else:
                success_fig = go.Figure()
                success_fig.add_annotation(
                    text="Success rate data not available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            # Processing Time Trends
            if 'processing_time' in df.columns:
                time_fig = px.box(
                    df,
                    x='category',
                    y='processing_time',
                    title='Processing Time Distribution by Category'
                )
                time_fig.update_layout(xaxis_title='Category', yaxis_title='Processing Time (days)')
            else:
                time_fig = go.Figure()
                time_fig.add_annotation(
                    text="Processing time data not available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            return volume_fig, category_fig, success_fig, time_fig
            
        except Exception as e:
            # Return error figures if something goes wrong
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"Error loading data: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return error_fig, error_fig, error_fig, error_fig
    
    return app 