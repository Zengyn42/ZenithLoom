/**
 * ZenithLoom Observability — Global Zustand Store
 * src/store/useStore.ts
 */

import { create } from 'zustand'
import type { AgentSnapshot, NodeStatus, ObservEvent, RawGraph } from '../types'

interface AgentGraphData {
  graph: RawGraph | null
  loading: boolean
  error: string | null
}

interface ObservStore {
  // Agent list and their runtime states
  agents: Record<string, AgentSnapshot>
  // Active agent tab
  activeAgent: string | null
  // Graph topology per agent
  graphData: Record<string, AgentGraphData>
  // WebSocket connection status
  wsStatus: 'connecting' | 'connected' | 'disconnected'
  // Recent events per agent (last 100)
  recentEvents: Record<string, ObservEvent[]>

  // Actions
  setActiveAgent: (name: string) => void
  handleInitMessage: (agents: AgentSnapshot[]) => void
  handleEvent: (evt: ObservEvent) => void
  setGraphData: (agent: string, graph: RawGraph | null, error?: string) => void
  setWsStatus: (status: 'connecting' | 'connected' | 'disconnected') => void
}

export const useStore = create<ObservStore>((set, get) => ({
  agents: {},
  activeAgent: null,
  graphData: {},
  wsStatus: 'disconnected',
  recentEvents: {},

  setActiveAgent: (name) => set({ activeAgent: name }),

  handleInitMessage: (agents) => {
    const agentMap: Record<string, AgentSnapshot> = {}
    for (const a of agents) {
      agentMap[a.name] = a
    }
    set((state) => ({
      agents: { ...state.agents, ...agentMap },
      // Set first online agent as active if nothing selected
      activeAgent:
        state.activeAgent ??
        agents.find((a) => a.online)?.name ??
        agents[0]?.name ??
        null,
    }))
  },

  handleEvent: (evt: ObservEvent) => {
    const { agent_name } = evt
    set((state) => {
      const agent: AgentSnapshot = state.agents[agent_name] ?? {
        name: agent_name,
        online: true,
        last_seen: evt.timestamp,
        last_seq: evt.seq,
        active_run_id: '',
        active_thread_id: '',
        node_states: {},
      }

      let nodeStates = { ...agent.node_states }

      if (evt.event_type === 'run_start') {
        // Reset all to idle on new run
        nodeStates = Object.fromEntries(
          Object.keys(nodeStates).map((k) => [k, 'idle' as NodeStatus])
        )
      } else if (evt.event_type === 'node_start') {
        const nid = (evt.payload.node as string) ?? evt.node_id
        nodeStates = { ...nodeStates, [nid]: 'running' }
      } else if (evt.event_type === 'node_end') {
        const nid = (evt.payload.node as string) ?? evt.node_id
        const hasError = Boolean(evt.payload.error)
        nodeStates = { ...nodeStates, [nid]: hasError ? 'error' : 'done' }
      } else if (evt.event_type === 'node_states_reset') {
        const incoming = evt.payload.node_states as Record<string, NodeStatus>
        nodeStates = { ...nodeStates, ...incoming }
      }

      const updated: AgentSnapshot = {
        ...agent,
        online: true,
        last_seen: evt.timestamp,
        last_seq: Math.max(agent.last_seq, evt.seq),
        active_run_id:
          evt.event_type === 'run_start' ? evt.run_id : agent.active_run_id,
        active_thread_id:
          evt.event_type === 'run_start' ? evt.thread_id : agent.active_thread_id,
        node_states: nodeStates,
      }

      const newActive =
        state.activeAgent === null ? agent_name : state.activeAgent

      // Append to recentEvents ring (last 100 per agent)
      const prevEvents = state.recentEvents[agent_name] ?? []
      const newEvents = [...prevEvents, evt]
      const trimmedEvents = newEvents.length > 100 ? newEvents.slice(newEvents.length - 100) : newEvents

      return {
        agents: { ...state.agents, [agent_name]: updated },
        activeAgent: newActive,
        recentEvents: { ...state.recentEvents, [agent_name]: trimmedEvents },
      }
    })
  },

  setGraphData: (agent, graph, error) =>
    set((state) => ({
      graphData: {
        ...state.graphData,
        [agent]: { graph, loading: false, error: error ?? null },
      },
    })),

  setWsStatus: (status) => set({ wsStatus: status }),
}))
