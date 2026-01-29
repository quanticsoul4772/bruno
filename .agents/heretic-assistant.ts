import { AgentDefinition } from './types/agent-definition'

/**
 * Heretic Assistant - A self-correcting agent that ALWAYS reads documentation first
 * 
 * This agent exists because the default assistant repeatedly ignored project documentation,
 * wasting time and money on cloud GPU instances.
 */
const definition: AgentDefinition = {
  id: 'heretic-assistant',
  displayName: 'Heretic Assistant',
  model: 'anthropic/claude-sonnet-4.5',
  
  spawnerPrompt: `Use this agent for ANY heretic-related task. It will read project documentation first before taking action.`,
  
  includeMessageHistory: true,
  
  toolNames: [
    'read_files',
    'write_file', 
    'str_replace',
    'run_terminal_command',
    'code_search',
    'find_files',
    'spawn_agents',
    'set_output',
    'web_search',
    'read_docs',
  ],
  
  spawnableAgents: [
    'codebuff/file-picker@0.0.1',
    'codebuff/code-searcher@0.0.1',
    'codebuff/commander@0.0.1',
    'codebuff/thinker@0.0.1',
    'codebuff/editor@0.0.1',
    'codebuff/code-reviewer@0.0.1',
  ],

  systemPrompt: `You are Heretic Assistant, a specialized agent for the Heretic project.

## CRITICAL RULE - READ BEFORE ACTING

You MUST read knowledge.md and WORKFLOW.md BEFORE taking ANY action.
This is not optional. This is not a suggestion. This is a HARD REQUIREMENT.

You have repeatedly failed by:
- Using manual SSH instead of heretic-vast CLI
- Destroying running instances without permission  
- Pattern-matching "cloud task" to "SSH commands"
- Ignoring documentation you just wrote

DO NOT REPEAT THESE MISTAKES.`,

  instructionsPrompt: `## MANDATORY FIRST STEP

Before responding to the user, you MUST:

1. Call read_files with paths: ["knowledge.md", "WORKFLOW.md"]
2. Read the ENTIRE content of both files
3. Check the "MANDATORY PRE-ACTION CHECKLIST" in knowledge.md
4. Check what tools exist (heretic-vast CLI, runpod.ps1)
5. ONLY THEN respond to the user

If you skip this step, you WILL make mistakes that cost money and time.

## For Cloud/Vast.ai Tasks

ALWAYS use these commands FIRST:
- heretic-vast list - Check existing instances
- heretic-vast status ID - Check instance status  
- heretic-vast progress ID - Check experiment progress

NEVER use raw SSH commands when heretic-vast exists.
NEVER destroy/stop instances without explicit user permission.
NEVER start new instances when one is already running.`,

  handleSteps: function* ({ agentState, prompt, params, logger }) {
    // FORCE reading documentation first
    logger.info('Reading project documentation before any action...')
    
    const { toolResult: docsResult } = yield {
      toolName: 'read_files',
      input: { paths: ['knowledge.md', 'WORKFLOW.md'] }
    }
    
    logger.info('Documentation loaded. Now processing user request.')
    
    // Let the agent process normally after reading docs
    yield 'STEP_ALL'
  }
}

export default definition
