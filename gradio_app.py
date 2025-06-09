import gradio as gr
import pandas as pd
import json
from agents import analyze_data_with_agent 
import io
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_data_and_prompt(file, prompt):
    """Process uploaded file and prompt using the data analysis agent."""
    try:
        if not file:
            return "Please upload a data file.", None, None
            
        if not prompt or prompt.strip() == "":
            return "Please enter an analysis prompt.", None, None

        # Read the uploaded file
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.name)
        elif file.name.endswith('.json'):
            df = pd.read_json(file.name)
        else:
            return "Error: Unsupported file format. Please upload CSV, Excel, or JSON files.", None, None

        # Clean column names
        df.columns = [str(col).strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        
        # Show data preview
        # data_preview = f"""
        # <div class="data-section">
        #     <h3>Data Preview</h3>
        #     <p><strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
        #     <p><strong>Columns:</strong> {', '.join(df.columns.tolist())}</p>
        #     {df.head().to_html(classes='table data-table', table_id='data-preview')}
        # </div>
        # """
        data_preview = f"""
<div></div>"""

        # Process with agent
        logger.info(f"Processing prompt: {prompt}")
        result = await analyze_data_with_agent(prompt, df)
        logger.info(f"Agent result type: {result.get('type')}")

        # Handle different result types
        if result["type"] == "error":
            error_html = f"""
            <div class="error-box">
                <h3>Error</h3>
                <p><strong>Message:</strong> {result['message']}</p>
                {f"<p><strong>Suggestions:</strong></p><ul>{''.join([f'<li>{s}</li>' for s in result.get('suggestions', [])])}</ul>" if result.get('suggestions') else ""}
            </div>
            """
            return data_preview + error_html, None, None
        
        elif result["type"] == "visualization":
            # Display the chart
            image_base64 = result.get("image")
            if image_base64:
                chart_html = f"""
                <div class="analysis-result">
                    <h3>Visualization Result</h3>
                    <p><strong>Chart Type:</strong> {result.get('chart_type', 'Unknown').title()}</p>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{image_base64}" class="chart-image">
                    </div>
                    <p><em>{result.get('message', 'Visualization created successfully')}</em></p>
                </div>
                """
                return data_preview + chart_html, None, None
            else:
                return data_preview + "<p>Error: Could not generate visualization</p>", None, None
        
        elif result["type"] == "statistical":
            # Format statistical results
            stat_html = f"""
            <div class="analysis-result">
                <h3>Statistical Analysis Results</h3>
                <div class="stat-output-box">
                    {result.get('data', 'No statistical results available')}
                </div>
                <p><em>{result.get('message', 'Statistical analysis completed')}</em></p>
            </div>
            """
            return data_preview + stat_html, None, None
        
        elif result["type"] == "transformation":
            # Return transformed data
            transformed_df = result.get("dataframe")
            if transformed_df is not None:
                # Create CSV for download
                csv_buffer = io.StringIO()
                transformed_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                # Create temporary file for download (Gradio handles temporary files for downloads)
                temp_file_name = "transformed_data.csv"
                with open(temp_file_name, 'w', encoding='utf-8') as f:
                    f.write(csv_data)
                
                transform_html = f"""
                <div class="analysis-result">
                    <h3>Data Transformation Results</h3>
                    <p><strong>Original Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
                    <p><strong>New Shape:</strong> {result.get('shape', 'Unknown')}</p>
                    <p><strong>New Columns:</strong> {', '.join(result.get('columns', []))}</p>
                    <div class="transformed-data-preview">
                        <h4>Preview of Transformed Data:</h4>
                        {result.get('preview', 'No preview available')}
                    </div>
                    <p><em>{result.get('message', 'Data transformation completed')}</em></p>
                    <p><strong>Download the transformed data using the button below.</strong></p>
                </div>
                """
                return data_preview + transform_html, temp_file_name, None
            else:
                return data_preview + "<p>Error: Could not retrieve transformed data</p>", None, None
        
        else:
            return data_preview + f"<p>Unknown result type: {result.get('type')}</p>", None, None

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        error_html = f"""
        <div class="error-box">
            <h3>Processing Error</h3>
            <p><strong>Error:</strong> {str(e)}</p>
            <p><strong>Please check:</strong></p>
            <ul>
                <li>File format is supported (CSV, Excel, JSON)</li>
                <li>File is not corrupted</li>
                <li>Prompt is clear and specific</li>
                <li>Ollama server is running</li>
            </ul>
        </div>
        """
        return error_html, None, None

def process_sync(file, prompt):
    """Synchronous wrapper for the async processing function."""
    try:
        # Check if an event loop is already running
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(process_data_and_prompt(file, prompt))
    except Exception as e:
        logger.error(f"Error in sync wrapper: {str(e)}")
        return f"Error: {str(e)}", None, None

def generate_preview(file):
    """Generate a preview of the uploaded file."""
    try:
        if not file:
            return "Please upload a data file to see preview."
            
        # Read the uploaded file
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.name)
        elif file.name.endswith('.json'):
            df = pd.read_json(file.name)
        else:
            return "Error: Unsupported file format. Please upload CSV, Excel, or JSON files."

        # Clean column names
        df.columns = [str(col).strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        
        # Show data preview
        data_preview = f"""
        <div class="data-section">
            <h3>📊 Data Preview</h3>
            <div class="data-stats">
                <span class="stat-badge">📏 {df.shape[0]} rows</span>
                <span class="stat-badge">📋 {df.shape[1]} columns</span>
            </div>
            <div class="columns-info">
                <strong>Columns:</strong> {', '.join(df.columns.tolist())}
            </div>
            <div class="table-container">
                {df.head(4).to_html(classes='table data-table', table_id='data-preview')}
            </div>
        </div>
        """
        return data_preview
    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}")
        return f"<div class='error-box'>Error generating preview: {str(e)}</div>"

# Sample prompts for different analysis types
sample_prompts = {
    "Visualization": [
        "Create a bar chart showing the distribution of categories",
        "Generate a line plot of sales over time",
        "Make a scatter plot of price vs quantity",
        "Show a histogram of customer ages",
        "Create a pie chart of market share by region"
    ],
    "Statistical Analysis": [
        "Calculate correlation matrix for all numeric columns",
        "Perform descriptive statistics analysis",
        "Compare means between different groups",
        "Find outliers in the dataset",
        "Calculate summary statistics by category"
    ],
    "Data Transformation": [
        "Filter data where sales > 1000 and add a profit column",
        "Group by category and calculate average values",
        "Remove duplicates and sort by date",
        "Create new columns based on existing ones",
        "Aggregate data by month and calculate totals"
    ]
}

# Create the Gradio interface
with gr.Blocks(
    title="Data Analysis Agent",
    theme=gr.themes.Soft(),
    css="""
    /* Main container */
    .gradio-container {
        max-width: 900px;
        margin: auto;
        padding: 20px;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: 600;
    }
    
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 1.1em;
        opacity: 0.9;
    }

    /* Accordion styling */
    .gr-accordion {
        margin-bottom: 20px !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-color-primary) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
        overflow: hidden !important;
    }
    
    .gr-accordion-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 15px 20px !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-accordion-header:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
        transform: translateY(-1px) !important;
    }
    
    .gr-accordion-content {
        background: var(--background-fill-secondary) !important;
        padding: 25px !important;
        border-top: 1px solid var(--border-color-primary) !important;
    }
    
    /* Special styling for example prompt accordions */
    .gr-accordion .gr-accordion {
        margin-bottom: 15px !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
    }
    
    .gr-accordion .gr-accordion .gr-accordion-header {
        background: var(--color-accent-soft) !important;
        color: var(--text-color-body) !important;
        padding: 12px 16px !important;
        font-size: 1em !important;
        font-weight: 500 !important;
    }
    
    .gr-accordion .gr-accordion .gr-accordion-header:hover {
        background: var(--color-accent) !important;
        color: white !important;
        transform: none !important;
    }
    
    .gr-accordion .gr-accordion .gr-accordion-content {
        background: var(--background-fill-primary) !important;
        padding: 15px !important;
    }

    /* Section styling (keeping for compatibility) */
    .section {
        background: var(--background-fill-secondary);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
        border: 1px solid var(--border-color-primary);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .section h2 {
        margin: 0 0 20px 0;
        color: var(--text-color-body);
        font-size: 1.4em;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* File upload styling */
    .upload-area {
        border: 2px dashed var(--border-color-accent);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: var(--background-fill-primary);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: var(--color-accent);
        background: var(--background-fill-hover);
    }

    /* Data preview styling */
    .data-section {
        background: var(--background-fill-primary);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid var(--border-color-primary);
        margin: 15px 0;
    }
    
    .data-section h3 {
        margin: 0 0 15px 0;
        color: var(--text-color-body);
        font-size: 1.2em;
    }
    
    .data-stats {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
        flex-wrap: wrap;
    }
    
    .stat-badge {
        background: var(--color-accent-soft);
        color: var(--text-color-body);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 500;
    }
    
    .columns-info {
        margin-bottom: 15px;
        padding: 10px;
        background: var(--background-fill-secondary);
        border-radius: 8px;
        font-size: 0.9em;
    }
    
    .table-container {
        overflow-x: auto;
        border-radius: 8px;
    }

    /* Table styling */
    .table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85em;
        background: var(--background-fill-primary);
    }
    
    .table th {
        background: var(--background-fill-secondary);
        color: var(--text-color-body);
        font-weight: 600;
        padding: 12px 8px;
        border: 1px solid var(--border-color-primary);
        text-align: left;
    }
    
    .table td {
        padding: 10px 8px;
        border: 1px solid var(--border-color-primary);
        color: var(--text-color-body);
    }
    
    .table tr:nth-child(even) {
        background: var(--background-fill-hover);
    }

    /* Prompt examples styling */
    .prompt-examples {
        display: grid;
        gap: 15px;
        margin-top: 15px;
    }
    
    .prompt-category {
        background: var(--background-fill-primary);
        border-radius: 8px;
        padding: 15px;
        border: 1px solid var(--border-color-primary);
    }
    
    .prompt-category h4 {
        margin: 0 0 10px 0;
        color: var(--text-color-body);
        font-size: 1em;
    }
    
    .prompt-buttons {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .prompt-btn {
        font-size: 0.8em !important;
        padding: 6px 12px !important;
        border-radius: 15px !important;
        background: var(--color-accent-soft) !important;
        color: var(--text-color-body) !important;
        border: 1px solid var(--border-color-accent) !important;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .prompt-btn:hover {
        background: var(--color-accent) !important;
        color: white !important;
    }

    /* Analysis results styling */
    .analysis-result {
        background: var(--background-fill-primary);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid var(--border-color-primary);
    }
    
    .analysis-result h3 {
        margin: 0 0 15px 0;
        color: var(--text-color-body);
    }

    /* Chart styling */
    .chart-container {
        text-align: center;
        margin: 20px 0;
        background: var(--background-fill-primary);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid var(--border-color-primary);
    }
    
    .chart-image {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Error styling */
    .error-box {
        background: #fee;
        border: 1px solid #fcc;
        color: #c33;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .error-box h3 {
        margin: 0 0 10px 0;
        color: #c33;
    }

    /* Button styling */
    .analyze-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 15px 30px !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .analyze-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 10px;
        }
        
        .main-header h1 {
            font-size: 2em;
        }
        
        .section {
            padding: 15px;
        }
        
        .data-stats {
            flex-direction: column;
        }
        
        .prompt-buttons {
            flex-direction: column;
        }
    }
    """
) as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🤖 Data Analysis Agent</h1>
        <p>Upload your data and get instant insights with AI-powered analysis</p>
    </div>
    """)
    
    # Step 1: File Upload
    with gr.Accordion("📁 Step 1: Upload Your Data", open=True):
        file_input = gr.File(
            label="Choose your data file (CSV, Excel, JSON)",
            file_types=[".csv", ".xlsx", ".xls", ".json"],
            type="filepath"
        )
    
    # Step 2: Data Preview
    with gr.Accordion("👀 Step 2: Data Preview", open=True):
        preview_output = gr.HTML(value="<p style='text-align: center; color: #888; padding: 40px;'>Upload a file to see data preview</p>")
    
    # Step 3: Analysis Prompt
    with gr.Accordion("💬 Step 3: Describe Your Analysis", open=True):
        prompt_input = gr.Textbox(
            label="What would you like to analyze?",
            placeholder="e.g., 'Create a bar chart showing sales by category' or 'Calculate correlation between price and quantity'",
            lines=3
        )
        
        # Example prompts in separate collapsible sections
        gr.HTML('<h4 style="margin: 20px 0 10px 0;">💡 Need inspiration? Try these examples:</h4>')
        
        with gr.Accordion("📊 Visualization Examples", open=False):
            for prompt in sample_prompts["Visualization"]:
                gr.Button(prompt, size="sm", elem_classes=["prompt-btn"]).click(
                    lambda p=prompt: p, inputs=[], outputs=prompt_input, queue=False
                )
        
        with gr.Accordion("📈 Statistical Analysis Examples", open=False):
            for prompt in sample_prompts["Statistical Analysis"]:
                gr.Button(prompt, size="sm", elem_classes=["prompt-btn"]).click(
                    lambda p=prompt: p, inputs=[], outputs=prompt_input, queue=False
                )
        
        with gr.Accordion("🔧 Data Transformation Examples", open=False):
            for prompt in sample_prompts["Data Transformation"]:
                gr.Button(prompt, size="sm", elem_classes=["prompt-btn"]).click(
                    lambda p=prompt: p, inputs=[], outputs=prompt_input, queue=False
                )
    
    # Step 4: Analysis Button
    with gr.Accordion("🚀 Step 4: Run Analysis", open=True):
        submit_btn = gr.Button("🚀 Analyze Data", variant="primary", size="lg", elem_classes=["analyze-btn"])
    
    # Step 5: Results
    with gr.Accordion("📊 Step 5: Analysis Results", open=True):
        output = gr.HTML(value="<p style='text-align: center; color: #888; padding: 40px;'>Click 'Analyze Data' to see results here</p>")
    
    # Step 6: Downloads
    with gr.Accordion("📥 Step 6: Downloads", open=True):
        download_output = gr.File(label="Transformed Data (if applicable)", visible=True)
        gr.HTML("<p style='color: #666; font-size: 0.9em;'>Download will appear here for data transformation results</p>")
    
    # Event handlers
    file_input.change(
        fn=generate_preview,
        inputs=[file_input],
        outputs=[preview_output]
    )
    
    submit_btn.click(
        fn=process_sync,
        inputs=[file_input, prompt_input],
        outputs=[output, download_output],
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )