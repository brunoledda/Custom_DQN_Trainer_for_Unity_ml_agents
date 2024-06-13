using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class TrainingAreaController : MonoBehaviour
{
    [System.Serializable]
    public class PlayerInfo
    {
        public RollerAgent Agent;
        [HideInInspector]
        public Vector3 StartingPos;
        [HideInInspector]
        public Rigidbody Rb;
    }
    public int MaxEnvironmentSteps = 5000;
    private SimpleMultiAgentGroup m_AgentGroup;
    public List<PlayerInfo> AgentsList = new List<PlayerInfo>();
    public Transform Target;
    private int resetTimer;
    // Start is called before the first frame update
    void Start()
    {
        m_AgentGroup = new SimpleMultiAgentGroup();   
        foreach (var item in AgentsList)
        {
            item.StartingPos = item.Agent.transform.localPosition;
            item.Rb = item.Agent.GetComponent<Rigidbody>();
            m_AgentGroup.RegisterAgent(item.Agent);
        }
        ResetScene();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        resetTimer += 1;
        bool agentOnPlatform = AgentOnPlatform();
        bool agentOnSwitch = AgentOnSwitch();

        if (resetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0)
        {
            m_AgentGroup.GroupEpisodeInterrupted();
            ResetScene();
        }
        foreach (var item in AgentsList)
        {
            if (item.Agent.transform.localPosition.y < 0)
            {
                m_AgentGroup.AddGroupReward(-0.1f);
                m_AgentGroup.GroupEpisodeInterrupted();
                ResetScene();
            }
            float distanceToTarget = Vector3.Distance(item.Agent.transform.localPosition, Target.localPosition);
            if (distanceToTarget < 1.42f)
            {
                m_AgentGroup.AddGroupReward(100.0f);
                m_AgentGroup.EndGroupEpisode();
                ResetScene();
            } 
        }
        /* if( countOnPlatform() == 0 || countOnSwitch() == 0)
        {
            m_AgentGroup.AddGroupReward(-0.005f);
        }
        if((countAroundPlatform() <= 2 && countAroundPlatform() > 0 ) || countAroundSwitch() == 1 )
        {
            m_AgentGroup.AddGroupReward(0.005f);
        }
         */

        if (agentOnPlatform && agentOnSwitch)
        {
            Debug.Log("IT HAPPENED!");
            m_AgentGroup.AddGroupReward(2.0f);
        }else if(agentOnPlatform || agentOnSwitch){
            Debug.Log("Solo una condizione è verificata");
            //m_AgentGroup.AddGroupReward(0.0001f/MaxEnvironmentSteps );
        }else{
            //Penalty for wasting time
            m_AgentGroup.AddGroupReward(-0.2f / MaxEnvironmentSteps); 
        }   
    }

    public void ResetScene(){
        resetTimer = 0;
        foreach (var item in AgentsList)
        {
            var pos = item.StartingPos;

            item.Agent.transform.localPosition = pos;
            item.Rb.velocity = Vector3.zero;
            item.Rb.angularVelocity = Vector3.zero;
        }
    }

    bool AgentOnPlatform()
    {
        bool isOnPlatform = false;
        foreach(var item in AgentsList)
        {
            if (item.Agent.isOnPlatform)
            {
                isOnPlatform = true;
            }
        }
        return isOnPlatform;
    }

    bool AgentOnSwitch()
    {
        bool isOnSwitch = false;
        foreach(var item in AgentsList)
        {
            if (item.Agent.isOnSwitch)
            {
                isOnSwitch = true;
            }
        }
        return isOnSwitch;    
    }

    int countOnPlatform()
    {
        int count = 0;
        foreach(var item in AgentsList)
        {
            if (item.Agent.isOnPlatform)
            {
                count++;
            }
        }
        return count;
    }
    int countOnSwitch()
    {
        int count = 0;
        foreach(var item in AgentsList)
        {
            if (item.Agent.isOnSwitch)
            {
                count++;
            }
        }
        return count;
    }

    int countAroundPlatform()
    {
        int count = 0;
        foreach(var item in AgentsList)
        {
            if (item.Agent.isAroundPlatform())
            {
                count++;
            }
        }
        return count;
    }
    int countAroundSwitch()
    {
        int count = 0;
        foreach(var item in AgentsList)
        {
            if (item.Agent.isAroundSwitch())
            {
                count++;
            }
        }
        return count;
    }
}

