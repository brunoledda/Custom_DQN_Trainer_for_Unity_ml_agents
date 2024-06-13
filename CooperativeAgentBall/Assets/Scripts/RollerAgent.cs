using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    public bool isOnPlatform;
    public bool isOnSwitch;
    
    
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
        isOnPlatform = false;
        isOnSwitch = false;

    }
    public Transform Target;
    public Transform platformFloor0;
    public Transform switchFloor0;
    
    /* public override void OnEpisodeBegin()
    {
       // If the Agent fell, zero its momentum
        if (this.transform.localPosition.y < 0 ||  Vector3.Distance(this.transform.localPosition, Target.localPosition)<1.42f)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3( 0, 0.5f, 0);
        }

        // Move the target to a new spot
        Target.localPosition = new Vector3(Random.value * 8 - 4,
                                           3.0f,
                                           3.24f);
        
    } */
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // Target, Agent, Platform and Switch positions
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        
        sensor.AddObservation(platformFloor0.localPosition);
        sensor.AddObservation(switchFloor0.localPosition);  

        // Agent velocity
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
        
        // Flag for platform and switch
        sensor.AddObservation(isOnPlatform ? 1.0f : 0.0f);
        sensor.AddObservation(isOnSwitch ? 1.0f : 0.0f);
    }

    public float forceMultiplier = 15.0f;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Actions, size = 2
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        if(isOnPlatform )
        {
            SetReward(0.5f);
        }
        else if (isOnSwitch)
        {
            SetReward(0.5f);
        }
        
      
        // Rewards
        //float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        // Reached target
        /* if (distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        } */

        // Fell off platform
        else if (this.transform.localPosition.y < 0)
        {
            SetReward(-0.4f);
            //EndEpisode(); The episode is already stopped by the TrainingAreaController.cs
        } 
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
    
    void FixedUpdate()
    {
        float raycastDistance = 5.0f;

        // Raycast per determinare se l'agente è sulla piattaforma
        RaycastHit hit;
        if (Physics.Raycast(transform.localPosition, Vector3.down, out hit, raycastDistance))
        {
            if (hit.collider.CompareTag("Platform"))
            {
                isOnPlatform = true;
            }
            else
            {
                isOnPlatform = false;
            }
        }
        else
        {
            isOnPlatform = false;
        }

        // Raycast per determinare se l'agente è sullo switch
        if (Physics.Raycast(transform.localPosition, Vector3.down, out hit, raycastDistance))
        {
            if (hit.collider.CompareTag("Switch"))
            {
                isOnSwitch = true;
            }
            else
            {
                isOnSwitch = false;
            }
        }
        else
        {
            isOnSwitch = false;
        }
    }
    
    public bool isAroundPlatform()
    {
        float distance = Vector3.Distance(this.transform.localPosition, platformFloor0.localPosition);
    
        float radius = 1.8f; 
        bool isAround = distance <= radius;
        return isAround; 
    }

    
    public bool isAroundSwitch()
    {
        float distance = Vector3.Distance(this.transform.localPosition, switchFloor0.localPosition);
    
        float radius = 1.3f; 
        bool isAround = distance <= radius;
        return isAround; 
    }
}
