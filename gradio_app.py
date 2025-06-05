import gradio as gr
import pandas as pd
import json
from agents import analyze_data_with_agent
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

async def process_data_and_prompt(file, prompt):
    """Process uploaded file and prompt using the data analysis agent."""
    try:
        # Read the uploaded file
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        else:
            return "Error: Unsupported file format. Please upload CSV, Excel, or JSON files."

        # Clean column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        # Process with agent
        result = await analyze_data_with_agent(prompt, df)

        # Handle different result types
        if result["type"] == "error":
            return f"Error: {result['message']}", None
        
        elif result["type"] == "visualization":
            # Create visualization
            plt.figure(figsize=(10, 6))
            chart_config = result["config"]
            
            if chart_config["chart_type"] == "bar":
                sns.barplot(data=df, x=chart_config["x_axis"], y=chart_config["y_axis"])
            elif chart_config["chart_type"] == "line":
                sns.lineplot(data=df, x=chart_config["x_axis"], y=chart_config["y_axis"])
            elif chart_config["chart_type"] == "scatter":
                sns.scatterplot(data=df, x=chart_config["x_axis"], y=chart_config["y_axis"])
            elif chart_config["chart_type"] == "histogram":
                sns.histplot(data=df, x=chart_config["x_axis"])
            
            plt.title(chart_config["title"])
            plt.tight_layout()
            
            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # Convert to base64 for display
            img_str = base64.b64encode(buf.read()).decode()
            return f'<img src="data:image/png;base64,{img_str}" />', None
        
        elif result["type"] == "statistical":
            # Format statistical results
            if isinstance(result["data"], dict):
                return json.dumps(result["data"], indent=2), None
            else:
                return str(result["data"]), None
        
        elif result["type"] == "transformation":
            # Return transformed data as a downloadable file
            transformed_df = pd.DataFrame(result["data"])
            csv = transformed_df.to_csv(index=False)
            return "Data transformation complete. Click the download button below to get the transformed data.", csv
        
        else:
            return "Unknown result type", None

    except Exception as e:
        return f"Error processing data: {str(e)}", None

# Create the Gradio interface
with gr.Blocks(title="Data Analysis Agent") as demo:
    gr.Markdown("# Data Analysis Agent")
    gr.Markdown("Upload your data file and enter a prompt to analyze it.")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Data File (CSV, Excel, or JSON)")
            prompt_input = gr.Textbox(
                label="Analysis Prompt",
                placeholder="Enter your analysis request (e.g., 'Create a bar chart of sales by category')"
            )
            submit_btn = gr.Button("Analyze")
        
        with gr.Column():
            output = gr.HTML(label="Results")
            download_output = gr.File(label="Download Results")
    
    submit_btn.click(
        fn=process_data_and_prompt,
        inputs=[file_input, prompt_input],
        outputs=[output, download_output]
    )

if __name__ == "__main__":
    demo.launch() 