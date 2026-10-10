import { Component, type ErrorInfo, type ReactNode } from 'react';

/** Keeps a rendering error from blanking the whole page: shows the message and lets the user recover. */
export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Dashboard render error', error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="flex min-h-screen items-center justify-center bg-navy-900 p-6">
        <div className="card w-full max-w-lg p-6" role="alert">
          <h1 className="text-lg font-bold text-navy-900">Something went wrong on this screen</h1>
          <p className="mt-2 text-sm text-slate-600">Your saved projects are not affected. Try again, or reload the page.</p>
          <pre className="mt-3 max-h-40 overflow-auto rounded bg-slate-100 p-2 text-xs text-red-700">{this.state.error.message}</pre>
          <div className="mt-4 flex gap-2">
            <button className="btn-primary" onClick={() => this.setState({ error: null })}>
              Try again
            </button>
            <button className="btn-secondary" onClick={() => window.location.reload()}>
              Reload page
            </button>
          </div>
        </div>
      </div>
    );
  }
}
