# Emacs Claude Code Package Comparison

When integrating Claude Code with Emacs, there are two available packages to choose from. This guide compares their features and provides a recommendation.

## Available Packages

### manzaltu/claude-code-ide.el

**Repository**: https://github.com/manzaltu/claude-code-ide.el

**Strengths**:
- Uses the Model Context Protocol (MCP) for proper IDE integration with automatic project detection and session management
- Built-in diagnostic integration with Flycheck and advanced diff view with ediff integration
- Proper selection and buffer tracking for better context awareness where Claude Code automatically knows which file you're currently viewing in Emacs
- More lightweight with focused feature set
- Uses vterm for terminal integration with full color support

**Limitations**:
- Currently in early development
- Fewer customization options
- Limited documentation and fewer commands
- Requires Emacs 28.1+

### stevemolitor/claude-code.el

**Repository**: https://github.com/stevemolitor/claude-code.el

**Strengths**:
- Comprehensive feature set with seamless Emacs integration, multiple Claude instances for different projects, and extensive customization options
- Rich command set including transient menu, quick responses, desktop notifications, and hook system for CLI integration
- Supports both eat and vterm terminal backends with extensive customization for each
- Well-documented with extensive configuration examples
- Active development with regular updates
- Optional IDE integration with Monet

**Limitations**:
- More complex setup and configuration
- Requires Emacs 30.0+ (newer requirement)
- Larger codebase with more dependencies

## Recommendation

**Use stevemolitor/claude-code.el** for the following reasons:

1. **Maturity and Documentation**: More mature with comprehensive documentation and examples
2. **Feature Completeness**: Offers a complete workflow with multiple instances, notifications, hooks, and extensive customization
3. **Active Community**: Better maintained with more GitHub activity and user feedback
4. **Flexibility**: Supports multiple terminal backends and extensive configuration options

The stevemolitor/claude-code.el package provides a much richer integration experience compared to the simpler claude-code-ide.el, making it the better choice for a comprehensive Claude Code workflow in Emacs.
