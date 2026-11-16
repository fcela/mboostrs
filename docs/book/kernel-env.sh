# Source before starting MyST so every notebook kernel emits SVG charts,
# including notebooks that use Altair without importing mboostrs.plot.
book_kernel_startup="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/kernel-startup.py"
if [ "${PYTHONSTARTUP:-}" != "$book_kernel_startup" ]; then
  export MBOOSTRS_DOCS_PREVIOUS_PYTHONSTARTUP="${PYTHONSTARTUP:-}"
fi
export PYTHONSTARTUP="$book_kernel_startup"
export MBOOSTRS_SVG_RENDERER=1
unset book_kernel_startup
