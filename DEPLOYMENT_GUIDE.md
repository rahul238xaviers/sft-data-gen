# Model Deployment Guide

## Overview

This guide covers the complete process of deploying your fine-tuned model to Ollama for integration with the LIMA application.

## Prerequisites

### System Requirements
- **Memory**: Minimum 16GB RAM (32GB+ recommended for large models)
- **Storage**: 50GB+ free space for model files and intermediate artifacts
- **Python**: 3.9 or higher
- **Ollama**: Latest version installed and running

### Software Dependencies
```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve

# Verify installation
ollama list
```

### Python Dependencies
All required packages should already be installed from `requirements.txt`:
- transformers
- peft
- torch
- python-dotenv

## Deployment Process

### 1. Configure Environment

Create/update your `.env` file based on `.env.example`:

```bash
cp .env.example .env
# Edit .env with your specific configuration
```

Key configuration parameters:
- `BASE_MODEL`: The base model used for fine-tuning (must match training config)
- `MODEL_NAME`: Full model identifier with organization/repo name
- `OUTPUT_MODEL_NAME`: Name for the deployed model in Ollama
- `QUANTIZATION`: Compression level (Q4_K_M, Q5_K_M, Q8_0, or none)

### 2. Run Deployment Notebook

Open and execute `deploy_model.ipynb`:

```bash
# Open in Jupyter or VSCode
jupyter notebook deploy_model.ipynb
```

Execute cells sequentially:

1. **Cell 1**: Import dependencies and setup logging
2. **Cell 2**: Load configuration and validate environment
3. **Cell 3**: Load base model and LoRA adapters
4. **Cell 4**: Merge LoRA adapters into base model
5. **Cell 5**: Save merged model to disk
6. **Cell 6**: Convert to GGUF format
7. **Cell 7**: Quantize and import to Ollama
8. **Cell 8**: Validate deployment (optional)

### 3. Verify Deployment

Test the model directly:

```bash
ollama run lima-finetuned-model "What is life insurance?"
```

Expected output: Model should respond with relevant insurance information.

## Integration with LIMA

### Update LIMA Configuration

Edit LIMA's `.env` file:

```bash
# Local model configuration
LOCAL_MODEL_NAME=lima-finetuned-model
LOCAL_MODEL_URL=http://localhost:11434
LOCAL_MODEL_TYPE=ollama

# Or as fallback model
FALLBACK_MODEL_NAME=lima-finetuned-model
FALLBACK_MODEL_TYPE=local
```

### Test Integration

```python
# In LIMA environment
from src.llm.local_connector import LocalModelConnector
from src.llm.base_connector import LLMConfig, ModelType

config = LLMConfig(
    model_name="lima-finetuned-model",
    base_url="http://localhost:11434",
    model_type=ModelType.CHAT,
    temperature=0.7,
    max_tokens=2048
)

connector = LocalModelConnector(config)
response = await connector.chat("What are the benefits of life insurance?")
print(response.content)
```

## Quantization Guide

### Choosing Quantization Level

| Level | Size Reduction | Quality | Use Case |
|-------|---------------|---------|----------|
| **Q4_K_M** | ~75% | Good | **Recommended** - Best balance |
| **Q5_K_M** | ~60% | Better | Higher quality, more memory |
| **Q8_0** | ~50% | Best | Maximum quality, large size |
| **none** | 0% | Perfect | No compression (not recommended) |

### Example Configurations

**Development/Testing** (Q4_K_M):
```bash
QUANTIZATION=Q4_K_M
```

**Production** (Q5_K_M):
```bash
QUANTIZATION=Q5_K_M
```

**High-Accuracy Applications** (Q8_0):
```bash
QUANTIZATION=Q8_0
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Error
**Symptom**: Process killed during model loading
**Solution**: 
- Reduce model size or use a smaller base model
- Close other applications
- Use CPU offloading: `device_map="auto"`

#### 2. GGUF Conversion Fails
**Symptom**: `convert_hf_to_gguf.py` errors
**Solution**:
- Update llama.cpp: `cd llama.cpp && git pull`
- Check model format compatibility
- Ensure merged model is saved correctly

#### 3. Ollama Import Fails
**Symptom**: `ollama create` returns error
**Solution**:
- Verify Ollama is running: `ollama list`
- Check GGUF file integrity
- Review Modelfile syntax
- Check logs: `journalctl -u ollama -f` (Linux) or Console.app (macOS)

#### 4. Model Produces Poor Results
**Symptom**: Irrelevant or low-quality responses
**Solution**:
- Verify fine-tuning was successful
- Test with different prompts
- Try lower quantization (Q5_K_M or Q8_0)
- Review training data quality

### Debugging Commands

```bash
# Check Ollama status
ollama list

# Remove problematic model
ollama rm lima-finetuned-model

# Test Ollama connection
curl http://localhost:11434/api/tags

# Monitor system resources
htop  # or Activity Monitor on macOS

# Check model file size
du -sh merged-model/ *.gguf

# Validate GGUF file
file lima-finetuned-model-q4_k_m.gguf
```

## Performance Optimization

### Memory Management
- Use `device_map="auto"` for automatic GPU/CPU distribution
- Clear intermediate models: `del model; gc.collect()`
- Monitor memory: `torch.cuda.memory_summary()` (if using GPU)

### Inference Speed
- Lower quantization = faster inference
- Reduce context window in Modelfile: `PARAMETER num_ctx 2048`
- Use GPU acceleration if available

### Model Size
- Q4_K_M typically reduces size by 75%
- Remove merged model after GGUF conversion to save space
- Use `--max_shard_size` during model saving for better handling

## Production Checklist

- [ ] Environment variables properly configured
- [ ] Base model and fine-tuned adapters available
- [ ] Sufficient disk space (50GB+)
- [ ] Ollama installed and running
- [ ] Model successfully deployed to Ollama
- [ ] Test prompts return expected results
- [ ] LIMA configuration updated
- [ ] Integration tests passing
- [ ] Performance benchmarks acceptable
- [ ] Monitoring/logging configured
- [ ] Rollback plan documented

## Maintenance

### Regular Tasks
1. **Monitor Performance**: Track response quality and latency
2. **Update Models**: Periodically retrain with new data
3. **Clean Up**: Remove old model versions to free space
4. **Backup**: Keep copies of successful model configurations

### Version Management
```bash
# List deployed models
ollama list

# Tag models with versions
ollama tag lima-finetuned-model lima-finetuned-model:v1.0

# Remove old versions
ollama rm lima-finetuned-model:v0.9
```

## Additional Resources

- [LIMA Integration Guide](LIMA_INTEGRATION.private.md)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Review deployment logs in `deployment.log`
3. Consult LIMA documentation
4. Contact the ML engineering team

---

**Last Updated**: January 2026
**Version**: 1.0
