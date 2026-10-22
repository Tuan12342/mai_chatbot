from mai_agent.policy import assess_message

# Handoff fallback uses the same general semantic schema as the policy guard. Keeping a
# single tool prevents the two routing stages from assigning incompatible meanings.
HANDOFF_TOOLS = [assess_message]
