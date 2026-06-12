// ZenithLoom Observability — shared TypeScript types

export interface ObservEvent {
  v: number
  agent_name: string
  thread_id: string
  run_id: string
  checkpoint_ns: string
  node_id: string
  event_type:
    | 'run_start'
    | 'run_end'
    | 'node_start'
    | 'node_end'
    | 'state_update'
    | 'node_states_reset'
    | 'ping'
  payload: Record<string, unknown>
  timestamp: number
  seq: number
}

export type NodeStatus = 'idle' | 'running' | 'done' | 'error'

export interface AgentSnapshot {
  name: string
  online: boolean
  last_seen: number
  last_seq: number
  active_run_id: string
  active_thread_id: string
  node_states: Record<string, NodeStatus>
}

// Raw graph data from /api/graph/{agent}
export interface RawNodeSpec {
  id: string
  type?: string
  agent_dir?: string
  subgraph?: { nodes: RawNodeSpec[]; edges: RawEdgeSpec[] }
  [key: string]: unknown
}

export interface RawEdgeSpec {
  from: string
  to: string
  type?: string
  [key: string]: unknown
}

export interface RawGraph {
  nodes: RawNodeSpec[]
  edges: RawEdgeSpec[]
  entry?: string
  exit?: string
  state_schema?: string
}

export interface InitMessage {
  type: 'init'
  agents: AgentSnapshot[]
}
