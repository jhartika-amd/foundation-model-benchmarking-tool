import os
import json
import math
import time
import boto3
import logging
import requests
import pandas as pd
from datetime import datetime
from fmbench.scripts import constants
from fmbench.utils import count_tokens
from typing import Dict, Optional, List
from litellm import completion, token_counter
from litellm import completion, RateLimitError
#import litellm
#from litellm import acompletion
#import asyncio, os, traceback

from fmbench.scripts.stream_responses import get_response_stream

from fmbench.scripts.fmbench_predictor import (FMBenchPredictor,
                                               FMBenchPredictionResponse)

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomRestPredictor(FMBenchPredictor):
    """
    This is a custom rest predictor that does a POST request on the endpoint
    specified in the configuration file with custom headers, authentication parameters
    and the model_id. This rest predictor can be used with custom parameters. View an 
    example of the parameters passed in this config file: configs/byoe/config-byo-custom-rest-predictor.yml
    """
    def __init__(self,
                 endpoint_name: str,
                 inference_spec: Optional[Dict],
                 metadata: Optional[Dict]):
        try:
            """
            Initialize the endpoint name and the inference spec. The endpoint name here points to the
            endpoint name in the config file, which is the endpoint url to do a request.post on. The 
            inference spec contains the different auth, headers, inference parameters that are 
            passed into this script from the config file.
            """
            self._endpoint_name: str = endpoint_name
            self._inference_spec: Dict = inference_spec

            self._temperature = 0.1
            self._max_tokens = 100
            self._top_p = 0.9
            # not used for now but kept as placeholders for future
            self._stream = None
            self._start = None
            self._stop = None

            # Initilialize the use_boto3 parameter to "False". This parameter is used if the 
            # current version of litellm does not support the model of choice to benchmark. In 
            # this case, the bedrock converseAPI is used
            self._use_boto3 = False

            # no caching of responses since we want every inference
            # call to be independant
            self._caching = False

            if inference_spec:
                parameters: Optional[Dict] = inference_spec.get('parameters')
                if parameters:
                    self._temperature = parameters.get('temperature', self._temperature)
                    self._max_tokens = parameters.get('max_tokens', self._max_tokens)
                    self._top_p = parameters.get('top_p', self._top_p)
                    self._stream = inference_spec.get("stream", self._stream)
                    self._stop = inference_spec.get("stop_token", self._stop)
                    self._start = inference_spec.get("start_token", self._start)
                    self._use_boto3 = parameters.get("use_boto3", self._use_boto3)
            self._response_json = {}
            logger.info(f"__init__, _bedrock_model={self._bedrock_model}, self._pt_model_id={self._pt_model_id},"
                        f"_temperature={self._temperature} "
                        f"_max_tokens={self._max_tokens}, _top_p={self._top_p} "
                        f"_stream={self._stream}, _stop={self._stop}, _caching={self._caching} "
                        f"_use_boto3={self._use_boto3}")
        except Exception as e:
            logger.error(f"create_predictor, exception occured while creating predictor "
                         f"for endpoint_name={self._endpoint_name}, exception={e}")
        logger.info(f"_endpoint_name={self._endpoint_name}, _inference_spec={self._inference_spec}")

    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        # Initialize some variables, including the response, latency, streaming variables, prompt and completion tokens.
        response_json: Optional[Dict] = None
        response: Optional[str] = None
        latency: Optional[float] = None
        # Streaming can be enabled if the model is deployed on SageMaker or Bedrock
        TTFT: Optional[float] = None
        TPOT: Optional[float] = None
        TTLT: Optional[float] = None
        prompt_tokens: Optional[int] = None
        response_dict_from_streaming: Optional[Dict] = None

        completion_tokens: Optional[int] = None
        streaming: Optional[bool] = None
   
        # This is the generated text from the model prediction
        generated_text: Optional[str] = None
        
        try:
            model_id = self._inference_spec.get("model_id")
            logger.info(f"model_id:{model_id}")

            model = "hosted_vllm/" + model_id
                
            logger.info("Going to use the standard text generation messages format to get inferences using Litellm")
            messages = [{"content": payload['inputs'], "role": "user"}]

            st = time.perf_counter()
            response = completion(model=model,
                                  messages=messages,
                                  api_base=self._endpoint_name,
                                  temperature=self._temperature,
                                  max_tokens=self._max_tokens,
                                  top_p=self._top_p,
                                  caching=self._caching,
                                  stream=self._stream)
            
            # Extract latency in seconds
            latency = time.perf_counter() - st
                
            logger.info(f"stop token: {self._stop}, streaming: {self._stream}, "
                        f"response: {response}")
            # Get the response and the TTFT, TPOT, TTLT metrics if the streaming
            # for responses is set to true
            if self._stream is True:
                response_dict_from_streaming = get_response_stream(response,
                                                                   st,
                                                                   self._start,
                                                                   self._stop,
                                                                   is_sagemaker=False)
                TTFT = response_dict_from_streaming.get('TTFT')
                TPOT = response_dict_from_streaming.get('TPOT')
                TTLT = response_dict_from_streaming.get('TTLT')
                response = response_dict_from_streaming['response']
                self._response_json["generated_text"] = json.loads(response)[0].get('generated_text')
                # Getting in the total input and output tokens using token counter.
                # Streaming on liteLLM does not support prompt tokens and completion tokens 
                # in the invocation response format
                prompt_tokens = token_counter(model=self._endpoint_name, messages=messages)
                completion_tokens = token_counter(text=self._response_json["generated_text"])
                logger.info(f"streaming prompt token count: {prompt_tokens}, "
                            f"completion token count: {completion_tokens}, latency: {latency}")
                logger.info("Completed streaming for the current UUID, moving to the next prediction.")
            # If streaming is set to false, then get the response in the normal
            # without streaming format from LiteLLM
            else:
                # Iterate through the entire model response
                # Since we are not sending batched requests so we only expect a single completion
                for choice in response.choices:
                    # Extract the message and the message's content from LiteLLM
                    if choice.message and choice.message.content:
                        # Extract the response from the dict
                        self._response_json["generated_text"] = choice.message.content
             # Extract number of input and completion prompt tokens
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                # Extract latency in seconds
                latency = response._response_ms / 1000
                # If we get here, the call was successful, so we break out of the retry loop

        except Exception as e:
            logger.error(f"Unexpected error during prediction, endpoint_name={self._endpoint_name}, "
                         f"exception={e}")
            raise  # Re-raise unexpected exceptions

        return FMBenchPredictionResponse(response_json=self._response_json,
                                         latency=latency,
                                         time_to_first_token=TTFT,
                                         time_per_output_token=TPOT,
                                         time_to_last_token=TTLT,
                                         completion_tokens=completion_tokens,
                                         prompt_tokens=prompt_tokens)
        
    @property
    def endpoint_name(self) -> str:
        """The endpoint name property."""
        return self._endpoint_name

    # The rest ep is deployed on an instance that incurs hourly cost hence, the calculcate cost function
    # computes the cost of the experiment on an hourly basis. If your instance has a different pricing structure
    # modify this function.
    def calculate_cost(self,
                       instance_type: str,
                       instance_count: int,
                       pricing: Dict,
                       duration: float,
                       prompt_tokens: int,
                       completion_tokens: int) -> float:
        """Calculate the cost of each experiment run."""
        #experiment_cost: Optional[float] = None
        experiment_cost: Optional[float] = 0.5
        try:
            #instance_based_pricing = pricing['pricing']['instance_based']
            #hourly_rate = instance_based_pricing.get(instance_type, None)
            #logger.info(f"the hourly rate for running on {instance_type} is {hourly_rate}, instance_count={instance_count}")
            ## calculating the experiment cost for instance based pricing
            #instance_count = instance_count if instance_count else 1
            #experiment_cost = (hourly_rate / 3600) * duration * instance_count
            experiment_cost = 0.5
        except Exception as e:
            logger.error(f"exception occurred during experiment cost calculation, exception={e}")
        return experiment_cost
    
    def get_metrics(self,
                    start_time: datetime,
                    end_time: datetime,
                    period: int = 60) -> pd.DataFrame:
        # not implemented
        return None

    def shutdown(self) -> None:
        """Represents the function to shutdown the predictor
           cleanup the endpooint/container/other resources
        """
        return None
    
    @property
    def inference_parameters(self) -> Dict:
        """The inference parameters property."""
        return self._inference_spec.get("parameters")

    @property
    def platform_type(self) -> Dict:
        """The inference parameters property."""
        return constants.PLATFORM_EXTERNAL
    
def create_predictor(endpoint_name: str, inference_spec: Optional[Dict], metadata: Optional[Dict]):
    return CustomRestPredictor(endpoint_name, inference_spec, metadata)