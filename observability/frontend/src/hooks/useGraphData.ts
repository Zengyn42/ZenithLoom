/**
 * Fetch AgentGraph topology from /api/graph/{agent}
 * src/hooks/useGraphData.ts
 */

import { useEffect } from 'react'
import { useStore } from '../store/useStore'

export function useGraphData(agentName: string | null) {
  const { graphData, setGraphData } = useStore()

  useEffect(() => {
    if (!agentName) return
    if (graphData[agentName]?.graph || graphData[agentName]?.loading) return

    // Mark loading
    setGraphData(agentName, null)

    fetch(`/api/graph/${agentName}`)
      .then((r) => r.json())
      .then((d) => {
        if (d.error) {
          setGraphData(agentName, null, d.error as string)
        } else {
          setGraphData(agentName, d.graph)
        }
      })
      .catch((err: Error) => {
        setGraphData(agentName, null, err.message)
      })
  }, [agentName]) // eslint-disable-line react-hooks/exhaustive-deps

  return agentName ? graphData[agentName] : null
}
