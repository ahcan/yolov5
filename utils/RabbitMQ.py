import pika
import logging
import logging.config
import json
from pathlib import Path

class RabbitQueue:
    def __init__(self, user, passwd, ip, port=5672, vhost='/', log_config=''):
        currentPath = Path(__file__).parent.absolute()
        logging.config.dictConfig(log_config)
        self.logger = logging.getLogger("Rabbit")
        self.user=user
        self.passwd=passwd
        self.ip=ip
        self.port=port
        self.vhost=vhost

    def __connect__(self):
        credentials = pika.PlainCredentials(self.user, self.passwd)
        try:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.ip, self.port,self.vhost,credentials, heartbeat=600, blocked_connection_timeout=300))
            #self.logger.info("Rabbit connection: {0}".format(self.connection))
        except Exception as ex:
            self.logger.error("RabbitQueue connection was: {0}".format(str(ex)))
        self.channel = self.connection.channel()

    def push_queue(self,route_key, message):
        self.routing_key = route_key
        try:
            self.__connect__()
            self.channel.queue_declare(queue= self.routing_key)
        except Exception as ex:
            self.logger.error("RabbitQueue.push: {0}.".format(str(ex)))
            try:
                self.connection.close()
            except Exception as ex:
                pass
            return False
        self.channel.basic_publish(exchange='',
            routing_key = self.routing_key,
            properties=pika.BasicProperties(delivery_mode=2,),
            body = message)
        self.connection.close()
        return True

    def get_queue(self, no_ack=True):
        try:
            self.__connect__()
        except Exception as ex:
            self.logger.error("RabbitQueue.get: {0}".format(ex))
            try:
                self.connection.close()
            except Exception as ex:
                pass
        self.channel = self.connection.channel()
        result = []
        for i in range(self.queue.method.message_count):
            try:
                body = self.channel.basic_get(queue=self.routing_key, auto_ack=no_ack) # get queue basic with single queue
                result.append(body)
            except Exception as ex:
                self.logger.error("RabbitQueue get route: {0}".format(ex))
            #print type( body)
        self.connection.close()
        return result

    def push_exchange(self, exchange_name, routing_key, message):
        self.exchange_name = exchange_name
        try:
            self.__connect__()
            #self.logger.info("Channel:{0}".format(self.channel))
            self.channel.exchange_declare(exchange=self.exchange_name, exchange_type='direct', durable=True, auto_delete=False)
        except Exception as ex:
            self.logger.error("RabbitQueue.setup exchange: {0}.".format(str(ex)))
            try:
                self.connection.close()
            except Exception as ex:
                self.logger.error("RabbitQueue.closeexchange: {0}.".format(str(ex)))
                pass
            return False
        try:
            self.channel.basic_publish(exchange=self.exchange_name,
                routing_key = routing_key,
                properties=pika.BasicProperties(delivery_mode=2,),
                body = message)
        except Exception as ex:
            self.logger.error("RabbitQueue.pushexchange{0}".format(str(ex)))
        self.connection.close()
        return True
    
    def get_exchange(self, exchange_name, routing_key, queue_name=''):
        self.exchange_name = exchange_name
        try:
            self.__connect__()
            self.channel.exchange_declare(exchange=exchange_name, exchange_type='direct', durable=True)
            tmp = self.channel.queue_declare(queue=queue_name, exclusive=True)
            self.queue_name = tmp.method.queue
            self.channel.queue_bind(exchange=exchange_name, queue=self.queue_name, routing_key=routing_key)
            return True
        except Exception as ex:
            self.logger.error("RabbitQueue.setup get exchange: {0}".format(str(ex)))
            try:
                self.connection.close()
            except Exception as ex:
                self.logger.error("RabbitQueue.close exchange: {0}".format(str(ex)))
                pass
            return False
        