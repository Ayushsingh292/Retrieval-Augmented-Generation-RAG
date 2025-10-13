def ensure_ollama_model(model_name: str):
    """
    Check if an Ollama model exists locally; if not, pull it automatically.
    """
    import subprocess, typer
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        local = result.stdout or ""
        if model_name not in local:
            typer.echo(f"Model '{model_name}' not found locally — pulling it...")
            subprocess.run(["ollama", "pull", model_name], check=True)
            typer.echo(f"✅ Model '{model_name}' downloaded successfully.")
        else:
            typer.echo(f"✅ Model '{model_name}' already available locally.")
    except subprocess.CalledProcessError as e:
        typer.echo(f"⚠️ Failed to list or pull models: {e}")
    except Exception as e:
        typer.echo(f"⚠️ Ollama auto-pull skipped due to error: {e}")
