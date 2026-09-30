import { StrictMode, Component, ReactNode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";

// Global error boundary so crashes never show a black screen
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (this.state.error) {
      return (
        <div className="flex h-screen items-center justify-center bg-zinc-950 text-center p-8">
          <div className="space-y-4">
            <div className="text-red-400 text-lg font-semibold">MLPilot encountered a runtime error</div>
            <pre className="text-xs text-zinc-500 bg-zinc-900 p-4 rounded-lg text-left overflow-auto max-w-xl">
              {(this.state.error as Error).message}
            </pre>
            <button
              className="mt-4 px-4 py-2 bg-emerald-500/90 text-emerald-950 font-medium rounded-lg text-sm"
              onClick={() => { this.setState({ error: null }); window.location.reload(); }}
            >
              Reload App
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>
);
