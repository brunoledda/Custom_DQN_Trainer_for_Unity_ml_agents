using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlatformSwitch : MonoBehaviour
{
    private bool isOnSwitch = false;
    private Collider platformCollider;
    private Vector3 loweredPosition; // Posizione abbassata dello switch
    private Vector3 maxPlatformPosition;
    public float upperAmount = 2.36f;
    public float lowerAmount = 0.1f; // di quanto lo switch si abbassa
    public float raycastLength = 1.0f;
    private Vector3 initialPosition;
    private Vector3 platformPosition;
    public Transform platform;

    private void Start()
    {
        initialPosition = this.transform.localPosition;
        loweredPosition = initialPosition - Vector3.up * lowerAmount; // Calcolo la posizione abbassata
        platformPosition = platform.localPosition;
        maxPlatformPosition= platformPosition + Vector3.up* upperAmount;
    }

    private void FixedUpdate()
    {
        // Lancio un raycast verso l'alto per controllare se il personaggio è vicino allo switch
        if (Physics.Raycast(this.transform.position, Vector3.up, out RaycastHit hitInfo, raycastLength))
        {
            if (hitInfo.collider.CompareTag("RollerAgent"))
            {
                this.isOnSwitch = true;
                
                //if(transform.localPosition.y > loweredPosition.y)
                //{
                    UpperPlatform();
                //}
                 
            }
            else
            {
                isOnSwitch = false;
            }
        }
        else
        {
            isOnSwitch = false;
            ReturnToInitialPosition();
        }
    }
    
    public float platformSpeed = 2.5f;
    private void UpperPlatform()
    {
        
        //if (this.isOnSwitch )
        //{
            Vector3 newPosition = this.transform.localPosition - Vector3.up * lowerAmount * Time.deltaTime*platformSpeed;
            newPosition.y = Mathf.Max(newPosition.y, loweredPosition.y);
            this.transform.localPosition = newPosition;

            Vector3 platformUpperPosition = platform.localPosition + Vector3.up * upperAmount * Time.deltaTime;
            platformUpperPosition.y = Mathf.Min(platformUpperPosition.y, maxPlatformPosition.y);
            platform.localPosition = platformUpperPosition;
        //}
    }

    private void ReturnToInitialPosition()
    {
        // Riporta lo switch alla posizione iniziale se non lo è
        if (this.transform.localPosition != initialPosition)
        {
            Vector3 newPosition = Vector3.MoveTowards(transform.localPosition, initialPosition, Time.deltaTime);
            this.transform.localPosition = newPosition;
        }
        // Riporta la piattaforma alla posizione iniziale se non lo è
        if (platform.localPosition != platformPosition)
        {
            Vector3 newPlatformPosition = Vector3.MoveTowards(platform.localPosition, platformPosition, Time.deltaTime*2.5f);
            platform.localPosition = newPlatformPosition;
        }
    }
}