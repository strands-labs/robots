# Troubleshooting

!!! info "Coming Soon"
    This page is under active development.

## Common Issues

### Serial port not found

```bash
# List available ports
python -m serial.tools.list_ports
```

### Camera not detected

```bash
# List video devices
ls /dev/video*
```

### GR00T inference service won't start

Check that the Docker container is running:

```bash
docker ps | grep isaac-gr00t
```

## Getting Help

- [GitHub Issues](https://github.com/strands-labs/robots/issues)
- [Strands Agents Documentation](https://strandsagents.com)
