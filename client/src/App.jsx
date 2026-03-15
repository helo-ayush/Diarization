import { useState, useRef } from 'react';
import { Upload, FileAudio, Search, Target, Loader2, Tag, CheckCircle2, ChevronDown, ChevronUp, X, Code, UserCheck } from 'lucide-react';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';

export default function App() {
  const [activeTab, setActiveTab] = useState('save');
  const [file, setFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [saveResult, setSaveResult] = useState(null);
  const [saveError, setSaveError] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const [showRawJson, setShowRawJson] = useState(false);
  const fileInputRef = useRef(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [searchLimit, setSearchLimit] = useState(5);
  const [allEntries, setAllEntries] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [searchError, setSearchError] = useState('');
  const [selectedResult, setSelectedResult] = useState(null);

  // ---------- ENROLL AGENT PIPELINE ----------
  const [agentName, setAgentName] = useState('');
  const [agentFile, setAgentFile] = useState(null);
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [enrollResult, setEnrollResult] = useState(null);
  const [enrollError, setEnrollError] = useState('');
  const agentFileInputRef = useRef(null);

  const handleAgentFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setAgentFile(e.target.files[0]);
      setEnrollError('');
    }
  };

  const handleEnroll = async (e) => {
    e.preventDefault();
    if (!agentFile || !agentName.trim()) return;

    setIsEnrolling(true);
    setEnrollError('');
    setEnrollResult(null);

    const formData = new FormData();
    formData.append('name', agentName.trim());
    formData.append('voice_sample', agentFile);

    try {
      const res = await fetch(`${API_URL}/api/v1/agents/enroll`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (data.error || !data.success) throw new Error(data.detail || data.error || data.message || 'Failed to enroll agent');
      setEnrollResult(data);
      setAgentName('');
      setAgentFile(null);
    } catch (err) {
      setEnrollError(err.message || 'Failed to enroll agent');
    } finally {
      setIsEnrolling(false);
    }
  };

  // ---------- SAVE PIPELINE ----------

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setSaveError('');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setSaveError('');
    }
  };

  const handleSave = async () => {
    if (!file) return;

    setIsProcessing(true);
    setSaveError('');
    setSaveResult(null);

    const formData = new FormData();
    formData.append('call_recording', file);

    try {
      const res = await fetch(`${API_URL}/api/v1/transcriptions`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (data.error) throw new Error(data.error);
      setSaveResult(data);
    } catch (err) {
      setSaveError(err.message || 'Failed to process audio');
    } finally {
      setIsProcessing(false);
    }
  };

  // ---------- SEARCH PIPELINE ----------

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    setSearchError('');
    setSearchResults(null);

    try {
      const res = await fetch(`${API_URL}/api/v1/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: searchQuery.trim(), 
          limit: parseInt(searchLimit),
          all_entries: allEntries 
        }),
      });
      const data = await res.json();

      if (data.error) throw new Error(data.error);
      setSearchResults(data);
    } catch (err) {
      setSearchError(err.message || 'Search failed');
    } finally {
      setIsSearching(false);
    }
  };

  // ---------- UI HELPERS ----------

  const formatTranscript = (text) => {
    if (!text) return '';
    return text.split('\n').map((line, i) => {
      let className = "text-gray-700";
      if (line.startsWith('Agent:')) className = "speaker-agent";
      else if (line.startsWith('Client:')) className = "speaker-client";

      return <div key={i} className={`mb-1 ${className}`}>{line}</div>;
    });
  };

  return (
    <div className="app-container">


      <div className="tabs">
        <button
          className={`tab ${activeTab === 'save' ? 'active' : ''}`}
          onClick={() => setActiveTab('save')}
        >
          <Upload size={16} /> Process
        </button>
        <button
          className={`tab ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          <Search size={16} /> Search
        </button>
        <button
          className={`tab ${activeTab === 'agents' ? 'active' : ''}`}
          onClick={() => setActiveTab('agents')}
        >
          <UserCheck size={16} /> Agents
        </button>
      </div>

      <div className="card">
        {/* SAVE TAB */}
        {activeTab === 'save' && (
          <div className="panel animate-fade-in">
            <div
              className={`upload-zone ${isDragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={(e) => { e.preventDefault(); setIsDragOver(false); }}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="audio/*,video/mpeg,audio/mpeg,video/mp4,audio/mp4,.mp3,.mpeg,.mpg,.m4a,.aac,.wav,.ogg"
                className="hidden"
                style={{ display: 'none' }}
              />
              <Upload size={24} className="upload-icon" />
              <div className="upload-text">
                {file ? file.name : 'Click or drop audio'}
              </div>
              <div className="upload-subtext">
                {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : 'Any format supported'}
              </div>
            </div>

            <button
              className="btn btn-primary w-full mt-6"
              onClick={handleSave}
              disabled={!file || isProcessing}
            >
              {isProcessing ? (
                <><Loader2 size={16} className="spin" /> Processing...</>
              ) : (
                'Analyze Audio'
              )}
            </button>

            {saveError && <div className="error-box mt-6"><Target size={16} /> {saveError}</div>}

            {saveResult && (
              <div className="results-container mt-6 animate-slide-up">
                <div className="flex justify-between items-center mb-6 border-b pb-3">
                  <h3 className="text-sm font-semibold flex items-center gap-2">
                    <CheckCircle2 size={16} /> Complete
                  </h3>
                  <div className="text-xs text-gray-400">ID: {saveResult._id}</div>
                </div>

                <div className="meta-badges mb-6">
                  <span className="badge score">
                    ⭐ {saveResult.analysis?.satisfactionScore || 0}/10
                  </span>
                  {(saveResult.analysis?.tags || []).map(t => (
                    <span key={t} className="badge"><Tag size={12} /> {t}</span>
                  ))}
                  <span className="text-xs text-gray-500 ml-auto">
                    {saveResult.metrics?.totalWords} words • STT {saveResult.metrics?.processingMs?.stt}ms
                  </span>
                </div>

                <div className="mb-6">
                  <h4 className="text-xs uppercase font-semibold text-gray-500 mb-2">Summary</h4>
                  <p className="summary-text">
                    {saveResult.analysis?.summary}
                  </p>
                </div>

                <div className="transcript-container">
                  <h4 className="text-xs uppercase font-semibold text-gray-500 mb-4">Transcript</h4>
                  <div className="text-sm">
                    {formatTranscript(saveResult.transcript)}
                  </div>
                </div>

                <div className="mt-8 border-t pt-4">
                  <button
                    className="flex items-center gap-2 text-xs font-semibold text-gray-400 hover:text-gray-900 transition-colors uppercase tracking-widest"
                    onClick={() => setShowRawJson(!showRawJson)}
                  >
                    {showRawJson ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    {showRawJson ? "Hide Raw JSON" : "View Raw JSON"}
                  </button>

                  {showRawJson && (
                    <div className="mt-4 animate-fade-in">
                      <pre className="json-viewer">
                        {JSON.stringify(saveResult, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* SEARCH TAB */}
        {activeTab === 'search' && (
          <div className="panel animate-fade-in">
            <form onSubmit={handleSearch} className="search-form">
              <div className="search-input-group flex-1">
                <Search size={16} className="search-icon" />
                <input
                  autoFocus
                  type="text"
                  placeholder="Search transcripts..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  className="search-input"
                />
              </div>

              <div className="limit-input-group w-24">
                <input
                  type="number"
                  min="1" max="50"
                  disabled={allEntries}
                  value={searchLimit}
                  onChange={e => setSearchLimit(e.target.value)}
                  className={`search-input text-center ${allEntries ? 'opacity-50' : ''}`}
                  title="Limit"
                />
              </div>

              <div className="flex items-center gap-2 px-3 border-l border-gray-100">
                <input
                  type="checkbox"
                  id="allEntries"
                  checked={allEntries}
                  onChange={e => setAllEntries(e.target.checked)}
                  className="w-4 h-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                />
                <label htmlFor="allEntries" className="text-xs font-medium text-gray-600 cursor-pointer whitespace-nowrap">
                  All Entries
                </label>
              </div>

              <button
                type="submit"
                className="btn btn-primary"
                disabled={!searchQuery.trim() || isSearching}
              >
                {isSearching ? <Loader2 size={16} className="spin" /> : 'Search'}
              </button>
            </form>

            {searchError && <div className="error-box mt-4"><Target size={16} /> {searchError}</div>}

            {searchResults && (
              <div className="results-container mt-6">
                <div className="flex items-center justify-between mb-6 pb-3 border-b">
                  <h3 className="text-sm font-semibold">
                    {searchResults.totalResults} Result{searchResults.totalResults !== 1 ? 's' : ''}
                  </h3>
                  <div className="text-xs text-gray-400">
                    {searchResults.processingMs}ms
                  </div>
                </div>

                {searchResults.results?.length === 0 ? (
                  <div className="empty-state">
                    <Search size={24} className="mb-4 text-gray-400 mx-auto" />
                    <p>No results found.</p>
                  </div>
                ) : (
                  <div className="search-results-list">
                    {searchResults.results.map((r) => (
                      <div
                        key={r._id}
                        className="search-result-card interactive"
                        onClick={() => setSelectedResult(r)}
                      >
                        <div className="flex justify-between items-start mb-4">
                          <div className="text-sm font-medium truncate" title={r.filename}>
                            {r.filename}
                          </div>
                          <span className="text-xs text-gray-400">
                            {new Date(r.createdAt).toLocaleDateString()}
                          </span>
                        </div>

                        <p className="summary-text mb-4">
                          {r.summary}
                        </p>

                        <div className="meta-badges">
                          <span className="badge score">
                            {(r.score * 100).toFixed(1)}% Match
                          </span>
                          <span className="badge">
                            ⭐ {r.satisfactionScore || 0}/10
                          </span>
                          {(r.tags || []).slice(0, 3).map(t => (
                            <span key={t} className="badge"><Tag size={10} /> {t}</span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* AGENTS TAB */}
        {activeTab === 'agents' && (
          <div className="panel animate-fade-in">
            <div className="mb-6">
              <h3 className="text-sm font-semibold mb-2">Enroll New Agent</h3>
              <p className="text-xs text-gray-500">Record or upload a clear 5-15 second voice sample of the agent speaking.</p>
            </div>
            
            <form onSubmit={handleEnroll}>
              <div className="mb-4">
                <input
                  type="text"
                  placeholder="Agent Name (e.g., 'Agent Rahul')"
                  value={agentName}
                  onChange={e => setAgentName(e.target.value)}
                  className="search-input w-full"
                  required
                />
              </div>

              <div
                className={`upload-zone mb-2 ${isDragOver ? 'drag-over' : ''} ${agentFile ? 'has-file' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
                onDragLeave={(e) => { e.preventDefault(); setIsDragOver(false); }}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragOver(false);
                  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                    setAgentFile(e.dataTransfer.files[0]);
                    setEnrollError('');
                  }
                }}
                onClick={() => agentFileInputRef.current?.click()}
              >
                <input
                  type="file"
                  ref={agentFileInputRef}
                  onChange={handleAgentFileChange}
                  accept="audio/*,video/mpeg,audio/mpeg,video/mp4,audio/mp4,.mp3,.mpeg,.mpg,.m4a,.aac,.wav,.ogg"
                  className="hidden"
                  style={{ display: 'none' }}
                />
                <FileAudio size={24} className="upload-icon" />
                <div className="upload-text">
                  {agentFile ? agentFile.name : 'Select Agent Voice Sample'}
                </div>
              </div>

              <button
                type="submit"
                className="btn btn-primary w-full mt-6"
                disabled={!agentFile || !agentName.trim() || isEnrolling}
              >
                {isEnrolling ? (
                  <><Loader2 size={16} className="spin" /> Extracting Fingerprint...</>
                ) : (
                  'Enroll Agent'
                )}
              </button>
            </form>

            {enrollError && <div className="error-box mt-6"><Target size={16} /> {enrollError}</div>}
            
            {enrollResult && (
              <div className="results-container mt-6 animate-slide-up border border-green-200 bg-green-50 p-4 rounded-md flex items-center gap-3">
                <CheckCircle2 size={24} className="text-green-600" />
                <div>
                  <h4 className="text-sm font-semibold text-green-900">Enrollment Successful</h4>
                  <p className="text-xs text-green-700">{enrollResult.message}</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* MODAL FOR SEARCH RESULTS */}
      {selectedResult && (
        <div className="modal-overlay" onClick={() => setSelectedResult(null)}>
          <div className="modal-content animate-slide-up" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div className="flex items-center gap-2">
                <Code size={18} />
                <h3 className="text-sm font-semibold">Document Details</h3>
              </div>
              <button
                className="modal-close"
                onClick={() => setSelectedResult(null)}
              >
                <X size={20} />
              </button>
            </div>
            <div className="modal-body">
              <div className="mb-6">
                <div className="text-xs text-gray-500 uppercase font-bold mb-2">Filename</div>
                <div className="text-sm font-medium">{selectedResult.filename}</div>
              </div>
              <div className="text-xs text-gray-500 uppercase font-bold mb-2">Raw JSON Data</div>
              <pre className="json-viewer large">
                {JSON.stringify(selectedResult, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
