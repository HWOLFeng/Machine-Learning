# BGP

```
2.
BGP-4提供了一套新的机制来支持无类域间路由。这些机制包括对宣传**IP前缀**并消除了网络的概念BGP中的“类”。BGP-4还引入了允许**路由聚合**，包括自治系统路径的聚合。

*关于什么政策可以和什么不能成为的更完整的讨论BGP强制实施不在本文档范围内（但请参阅讨论BGP使用情况的随附文档[5]）。*

初始数据流是整个BGP路由表。增加的路由表更改时发送更新。BGP不需要定期刷新整个BGP路由表。
3. 
因此，BGP扬声器必须保留整个BGP路由的当前版本连接期间所有对等方的表。
定期发送KeepAlive消息，以确保连接。发送通知消息以响应错误或特殊条件。
如果连接遇到错误条件，发送通知消息

如果特定的AS具有多个BGP发言人并正在提供转接其他AS的服务，则必须注意确保一致AS内部路由视图。
内部的一致看法AS的路由由内部路由协议提供。

可以通过以下方式提供AS外部路由的一致视图：
- 让AS中的所有BGP发言人保持直接BGP连接彼此。BGP发言人使用一组通用策略达成协议，确定哪些边界路由器将用作AS外部特定网络的退出/进入点。
（必须注意确保内部路由器均已更新中转信息在BGP发言人向其他AS宣布过渡服务是被提供。）

同样，不同的AS被称为外部对等体，而相同的AS可以描述为内部对等体。

3.1
路由被定义为将目的地与路径属性配对的信息。

在UPDATE中在一对BGP发言者之间通告路由消息：目标是其IP地址为网络层可达性信息（NLRI）

如果BGP发言人选择发布路由，则可能会添加到或修改路径的**路径属性**，然后再将其发布给同行。

给定的BGP讲话者可以通过三种方法指示一条路线已从服务中撤出：
a）表示先前目的地的IP前缀可以在 WITHDRAWN ROUTES字段中通告发布的路由在UPDATE消息中，从而将关联的路由标记为不再可用（WITHDRAWN）

b）具有相同网络层可达性的替代路由信息可以被宣传（替代）

c）BGP发言者-可以关闭BGP发言者连接，其中从服务中隐式删除对演讲者互相宣传（主动删除）

3.2 
路由存储在路由信息库（RIB）中，即Adj-RIBs-In，Loc-RIB和Adj-RIBs-Out。
路线将会向其他BGP发言人宣传Adj-RIB-Out。
本地BGP发言人将使用的路由必须在Loc-RIB中存在，并且每个这些的下一跳本地BGP发言人的转发中必须存在路由信息库；
以及从其他BGP收到的路由Adj-RIBs-In中有扬声器。

a）Adj-RIBs-In：存储的路由信息是通过学习入站信息。这些内容表示可作为决策过程输入的路径。

b）Loc-RIB：包含BGP发言人通过将其本地策略 应用于其Adj-RIBs-In中包含的路由信息 而选择的本地路由信息。

c）Adj-RIBs-Out：存储本地BGP发言人已选择向其对等方通告的信息。 存储在Adj-RIBs-Out中的路由信息将携带在本地BGP发言人的UPDATE消息中，并通告给其对等方。

总之，Adj-RIBs-In包含未处理的路由信息已经由其对等方通告给本地BGP发言人的内容；的Loc-RIB包含由本地BGP选择的路由发言人的决策程序；和Adj-RIBs-Out整理路线通过本地演讲者的广告向特定的同伴广告 UPDATE消息。

4.  Message Formats
4.1 Message Header Format
到时候查文档看，暂时没什么用
Length: at least 19 and no greater than 4096
Type:
This 1-octet unsigned integer indicates the type code of the message.  The following type codes are defined:
1 - OPEN
2 - UPDATE
3 - NOTIFICATION
4 - KEEPALIVE

4.2 OPEN Message Format
建立传输协议连接后，第一个双方发送的消息是OPEN消息。
- Version: 1-octet unsigned integer indicates the protocol version
- My Autonomous System: This 2-octet unsigned integer indicates the Autonomous System number of the sender.
- Hold Time: This 2-octet unsigned integer indicates the number of seconds that the sender proposes for the value of the Hold Timer. a BGP speaker MUST calculate the value of the Hold Timer by using the **smaller** of its **configured** Hold Time and the Hold Time **received in the OPEN message**.
- BGP Identifier: The value of the BGP Identifier is determined on startup and is the same for every local interface and every BGP peer.

Authentication Code:
Authentication Data: Message Length = 29 + Authentication Data Length.

The minimum length of the OPEN message is 29 octets (including message header).

4.3 UPDATE Message Format
The information in the UPDATE packet can be used to **construct a graph describing the relationships of the various Autonomous Systems**.

An UPDATE message is used to advertise a single feasible route to a peer, or to withdraw multiple unfeasible routes from service

The UPDATE message always includes the fixed-size BGP header, and can optionally include the other fields as shown below:
+-----------------------------------------------------+
|   Unfeasible Routes Length (2 octets)               |
+-----------------------------------------------------+
|  Withdrawn Routes (variable)                        |
+-----------------------------------------------------+
|   Total Path Attribute Length (2 octets)            |
+-----------------------------------------------------+
|    Path Attributes (variable)                       |
+-----------------------------------------------------+
|   Network Layer Reachability Information (variable) |
+-----------------------------------------------------+

- Unfeasible Routes Length: indicates the total length of the Withdrawn Routes field in octets.
- Withdrawn Routes: Each IP address prefix is encoded as a 2-tuple of the form <length, prefix>.
- Total Path Attribute Length: indicates the total length of the Path Attributes field in octets.
- Path Attributes: A variable length sequence of path attributes is present in every UPDATE. Each path attribute is a triple **<attribute type, attribute length, attribute value>** of variable length. 
0                   1
0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Attr. Flags  |Attr. Type Code|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
-- Attribute Type is a two-octet field that consists of the Attribute Flags octet followed by the Attribute Type Code octet.
（详情请看文档）
-- Attribute Type Code.
a) ORIGIN (Type Code 1):IGP、EGP、INCOMPLETE（Network Layer Reachability Information learned by some other means）
b) AS_PATH (Type Code 2): Each AS path segment is represented by a triple <path segment type, path segment length, path segment value>.
-- AS_SET(1): unordered set of ASs
-- AS_SEQUENCE(2): ordered set of ASs
c) NEXT_HOP (Type Code 3): defines **the IP address of the border router** that should be used as the next hop to the destinations listed in the Network Layer Reachability field of the UPDATE message.
d) MULTI_EXIT_DISC (Type Code 4):
e) LOCAL_PREF (Type Code 5):
f) ATOMIC_AGGREGATE (Type Code 6)
g) AGGREGATOR (Type Code 7)
- Network Layer Reachability Information:
```

