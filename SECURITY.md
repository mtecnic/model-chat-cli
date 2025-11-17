# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |

## Security Features

### Input Validation

All user inputs are validated using `utils/validation.py`:

- **Message Length Limits**: Max 100k characters to prevent memory exhaustion
- **System Prompt Limits**: Max 10k characters
- **Filename Sanitization**: Prevents path traversal attacks
- **URL Validation**: Blocks malicious protocols and XSS attempts
- **Port Validation**: Ensures valid port ranges (1-65535)

### File Safety

- **Disk Space Checks**: Requires minimum 1MB free before saving
- **File Size Limits**: Max 50MB for conversation files
- **Extension Blocking**: Dangerous executables (.exe, .dll, .sh, .bat) blocked
- **Path Traversal Protection**: `../` patterns blocked in all file operations
- **Symlink Detection**: Symlinks rejected to prevent attacks
- **Corrupted File Handling**: Graceful degradation for malformed JSON

### Network Security

- **Timeout Protection**: 30s scan timeout, 2s per probe
- **Rate Limiting**: Max 50 concurrent probes (semaphore)
- **Retry Logic**: Exponential backoff (2s→16s) prevents DoS
- **Connection Limits**: httpx connection pooling configured

### Memory Protection

- **Bounded History**: Max 1000 messages in memory
- **Display Limits**: Max 100 messages rendered
- **Context Window Management**: Auto-trimming prevents token overflow
- **Conversation Limits**: Max 10k messages per saved conversation

### Data Protection

- **No Credential Storage**: Never stores API keys or passwords
- **Local-Only Data**: All data stored in user's home directory
- **JSON Encoding**: UTF-8 with proper escaping
- **Error Logging**: Sensitive data excluded from logs

## Dependency Security

All dependencies scanned with `pip-audit`:

```bash
pip-audit --requirement requirements.txt
```

**Last Scan**: 2025-01-16
**Result**: ✅ No known vulnerabilities

| Dependency | Version | Status |
|------------|---------|--------|
| rich | 14.2.0 | ✅ Safe |
| httpx | 0.28.1 | ✅ Safe |
| prompt-toolkit | 3.0.52 | ✅ Safe |
| pytest | 9.0.1 | ✅ Safe |

## Reporting a Vulnerability

If you discover a security vulnerability, please:

1. **DO NOT** open a public issue
2. Email the maintainer with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and provide a fix within 7 days for critical issues.

## Security Best Practices for Users

### When Running the CLI:

1. **Use Virtual Environment**: Always run in a Python virtual environment
2. **Review Code**: Audit any code before running (this is open source!)
3. **Network Isolation**: Run on trusted networks only
4. **Local Models**: Use local LLMs for sensitive data
5. **Regular Updates**: Keep dependencies up to date

### Environment Variables:

```bash
# Restrict scan scope
export MODEL_CHAT_COMMON_PORTS="11434,1234"
export MODEL_CHAT_SCAN_TIMEOUT="15.0"

# Limit memory usage
export MODEL_CHAT_MAX_HISTORY="500"
export MODEL_CHAT_MAX_DISPLAY="50"
```

## Security Audit History

### 2025-01-16: Comprehensive Security Audit

**Auditor**: Eddy the Crusher
**Score**: B+ (83/100)

**Critical Fixes Implemented**:
- ✅ Command injection prevention (/theme typo)
- ✅ DoS protection (network scanner timeouts)
- ✅ Race condition fixes (theme switching)
- ✅ Memory exhaustion prevention (bounded history)

**High Priority Fixes Implemented**:
- ✅ Retry logic with exponential backoff
- ✅ Token estimation extraction (DRY)
- ✅ Configuration management system
- ✅ Input validation framework
- ✅ File safety checks

**Medium Priority Fixes Implemented**:
- ✅ Conversation save/load with validation
- ✅ Context window management
- ✅ Automated test suite (60%+ coverage)

## Testing

Run security-focused tests:

```bash
# All tests
pytest

# Security-specific tests
pytest tests/test_validation.py -v

# With coverage
pytest --cov=utils --cov-report=html
```

## Known Limitations

1. **Local Network Only**: Designed for local model servers
2. **No Authentication**: Assumes trusted local environment
3. **No Encryption**: Data stored unencrypted in home directory
4. **Python Execution**: Running Python code always carries inherent risks

## Security Checklist for Contributors

- [ ] Input validation for all user inputs
- [ ] Path traversal checks for file operations
- [ ] Timeout configuration for network calls
- [ ] Error handling that doesn't leak sensitive data
- [ ] No hardcoded credentials or secrets
- [ ] Dependencies audited with `pip-audit`
- [ ] Tests covering security-critical paths
- [ ] Documentation of security implications

## License

This security policy is part of the Model Chat CLI project and follows the same license.
