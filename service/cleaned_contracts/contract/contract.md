Project Code: LPRO-2022-21

# Statement of Work for provision of services to LPRO Holdings IN

# MCA Systems Portugal (Spain Branch)

**property name supplied.** 

| Customer | or | LPRO Parts Holding NV a company incorporated in Belgium with a |
| --- | --- | --- |
| LPRO: |  | registered number of 1217890123 |
| MCA: |  | MCA Systems Portugal (Spain branch) having its principal place of |
|  |  | business in Spain |
| Agreement |  | The Master Services Agreement executed between Customer and MCA |
|  |  | dated 01-02-2020 |
| SOW | Effective | 13/JUN/2022 |
| Date |  |  |
| S O W | E x p i r y | 31/DEC/2022 |
| Date |  |  |

This Statement of Work ("SOW") sets out the services to be provided to Customer by MCA pursuant to the Agreement, and this SOW shall be considered an integral part of that Agreement.

# 1. **Term of the SOW**

The term of this SOW shall, notwithstanding the date of signature hereof, commence as of the SOW Effective Date set out above and unless terminated earlier in accordance with the Agreement shall expire and terminate on the SOW Expiry Date.

# 2. **Scope and purpose of Services**

### **2.1. Scope**

The Scope of this SOW is the provision of consulting and implementation services to support the LPRO Search infrastructure. This includes advisory services, solution design options, help with the technology selection and implementation of the parts search functionality. An MCA Delivery Manager will supervise the MCA staff activities. All projects and services shall be agreed in writing between MCA and the Customer. The agreed, project scope shall be executed by the team comprising of personnel from the Customer and MCA and tracked in a project tracking tool either in Customer or MCA or in both of their internal systems, as agreed in writing.

The services provided by MCA will follow evolutionary approach with parallel execution whenever possible. The goal will be to optimise the total cost of ownership by considering both build and buy solution as well as a hybrid one. The key architectural concept is placing a separate search infrastructure in the landscape. All search queries and the respective constrains will be served by that infrastructure. The relevant data will be synchronised to the search infrastructure from the main information stores such as parts data, PIM or Link.

The concept is visualised in the diagram below:

The diagram visualises the search infrastructure, the aggregated data sources, core that handles the data structures, queries and restrictions. On top the search infrastructure provides APIs for integration with various applications as well as its own user interface.

14/12/2015 16:5407/04/2015 10:29:006/29/2009 4:22:00 PM**Error!** No

**property name supplied.** 

The diagram also depicts the possible areas for technology selection as the core of the search infrastructure could be acquired as commercial product or implemented on top of the common search engines like Elasticsearch.

As the technology selection is a long process the initial phase is separated in two streams:

- a. Build stream addressing all the clear elements like infrastructure, connection to the underlying data sources, clarification of data models, etc..
- b. Discover stream addressing the technology selection and fining the optimal solution for LPRO

### 2.1.1. *Execution framework*

Working with the relevant Customer staff, specific tasks will be assigned to MCA based on an agreed written plan to ensure each stream of work has clear goals and deliverables. Milestones may be agreed in writing by the parties from time to time as part of the iteration (mission board) planning . Deliverables are planned at the start of iteration/mission board (each iteration is 2 months) for the squad. Each of the deliverables planned during the iteration are to be delivered in a sprint and measured in story points. Each deliverable must be mutually agreed for delivery and should have agreed value in story points.

The client's PO along with the Functional analyst and the technical lead will refine the user stories with the team and agree to the story points the team proposes for the user stories to be implemented end to end (development and testing).

The squads are measured against the say-do ratio for each sprint. The client and MCA agree that there is a 10% minimum tolerance level for the squad and the squad will make continuous steps to improve the tolerance level. If the squad is not able to deliver within the set tolerance level, Customer and MCA will agree on reasonable mitigation through which MCA will make additional efforts to achieve the minimum tolerance level.

Any changes to the deliverable, including but not limited to requirements changes or clarifications, UX or visual design changes, changes to applicable technical specifications or standards upon starting the implementation are considered new deliverables.

MCA reserves the right to charge the Client for the effort already expended if Client requests to suspend or cancel the work on previously approved deliverable.

### 2.1.2. *General assumptions*

The main approach for both streams will be timeboxed work with the aim to cover the whole space in terms of components and interactions. The implementation alternatives will be reasonably detailed for the steering decisions which will be checked on a regular basis with the Client.

#### 2.1.3. *Team composition and retention*

As part of the team composition MCA will provide a product owner role and will work together with the client to establish an internal person that can take over that in the long term.

In the first 6 weeks of the engagement the foundation work will be performed by a smaller team to ensure proper landing of the rest of the engineers. Once the foundation work is completed together with the technology selection the team will start taking on the search features in order provided by the customer.

**property name supplied.** 

#### The MCA Search team will be constructed specifically for the search infrastructure implementation and evolution.

#### 2.1.4. *Deployment model*

As part of the technical description of the platform, a deployment strategy should be illustrated. It should be in sync with the overall cloud approach of the customer.

#### **2.2. Services**

The parties agree that the Services provided by MCA under this SOW shall be considered as services and the SOW as a service agreement. The Services shall not be considered as works and the SOW shall not be considered a work contract. No specific results shall be owned solely by MCA such that the provision of Services to deliver the outcome shall be a joint responsibility of the parties. In delivering the Services, MCA will use all the skill, care and diligence reasonably to be expected of a qualified and competent IT service provider experienced in undertaking similar services. To the extent deliverables are mentioned in this SOW, this is only for the purpose of describing the context in which the Services are provided.

#### **2.3. Intended Deliverables**

As outcome of the exercise the aim is to produce an architecture approach, technology selection and functional implementation of the search infrastructure. Gradually different search functions will be added such as "category search", "reference search", etc.

The implementation will include documented APIs as well as basic user interface.

# **2.4. Acceptance**

#### Process:

MCA shall notify the Customer that the deliverable has been completed by changing status of the corresponding JIRA ticket to resolved.

The team will conduct demo meetings where delivered user stories are demonstrated to Product Owner demonstrating that DoD provided in Appendix A are reached.

MCA will execute applicable tests defined in the user story acceptance criteria or other applicable specifications and demonstrate test results prior to acceptance.

Customer will review the deliverables upon deployment to a mutually agreed acceptance environment. Customer shall accept or reject deliverables based on specific acceptance criteria defined for each deliverable within ten (10) business days (Acceptance Period).

Customer will notify acceptance of the deliverables in accordance with the JIRA workflow, details to be agreed. A deliverable will be considered automatically accepted if not rejected within the Acceptance Period.

#### **2.5. Professionals**

MCA will provide the Services using its Professionals, taking into account the complexity of the Services and timeline for deliverables, as may be agreed from time to time

The Professionals shall in no event be seen as employees, agents or (legal) representatives of the Customer.

# **Title**: **Statement of Work for provision of services Confidential**  14/12/2015 16:5407/04/2015 10:29:006/29/2009 4:22:00 PM**Error!** No

#### **property name supplied.**

The profile, location and/or number of MCA Personnel assigned to this Statement of Work may be separately agreed or amended in writing or via exchange of email in a resource plan (the "**Resource Plan**"). The initial Resource Plan is set out in section 6 below and might be reviewed by the Customer and MCA in case of unexpected discovery.

If the Services are performed by a Professional who is an Employee of MCA, the Customer acknowledges that the employer's authority over the Professional who is an Employee of MCA lies exclusively with MCA. The Customer shall at all times refrain from exercising any (part of the) employer's authority over the Professional who is an Employee of MCA, or give him instructions which are typical for employers and may result in prohibited putting at the disposal of personnel.

The Customer expressly recognises that it has no right or power or authority whatsoever over the Professionals who are employees of MCA without prejudice, however, to the right to provide guidelines and instructions regarding the Services to be performed under this SOW. More particularly the Customer may issue guidelines and reasonable instructions with regard to the following without impacting the Professionals' position as employees of MCA:

- the legal obligations on well-being at work, including safety guidelines;
- the further specification and timeline of the Services detailed in this SOW;
- communication processes to be used between MCA and the Customer to exchange information and data relevant to the Services;
- use of Customer's property, working space, materials and equipment used for the provision of Services;
- access to Customer premises;
- working hours as applicable at the workplace of the Customer without directly imposing any working hours or working schedule on the Professional who is an Employee of MCA.

The parties acknowledge that the guidelines described above do not harm or contradict MCA's position as only employer of the Professionals who are Employees of MCA.

MCA is exclusively responsible for the payment of the salary and the employee ("social ") benefits to the personnel. The salary and employee benefits will be paid in accordance with the agreements between MCA and the personnel.

If the Services are performed by a Professional who is an Independent Contractor with whom MCA has entered into a services contract, the Customer acknowledges that the Independent Contractor offers his services on an independent business to business basis. The Customer shall at all times refrain from exercising any employer's authority over the Independent Contractor or give him/her instructions which are typical for employers and may result in the re-qualification of the mutual relationships as an employment contract between MCA and/or the Customer on the one hand and the Independent Contractor on the other hand. The Customer shall limit any guidelines it gives directly to the Independent Contractor to:

- safety guidelines applying to any visitor.
- guidelines related to the opening hours of the Customer's offices.
- guidelines related to the Customer's property, working space, materials and equipment used for the provision of Services; and
- the exchange of information and follow-up of the activities performed on the Customer's premises, without interfering in the way in which the Independent Contractor organizes his activities in the framework of his services contract with MCA.

14/12/2015 16:5407/04/2015 10:29:006/29/2009 4:22:00 PM**Error!** No

If the Customer has any remarks regarding the performance of the Professionals who are Employees of MCA or Independent Contractors with whom MCA has entered into a services contract, i.e. if the Customer is of the opinion at any time during the term of the Agreement that the Professional is not performing the Services satisfactorily, is not showing the level of skills that is required for the performance of Services or is otherwise not acting as may be expected from a professional, the Customer will inform MCA as soon as reasonably possible, not the Professional directly. MCA will take the necessary measures – in mutual consultation with the Customer – towards the Professionals, and if necessary, replace the Professional, in accordance with article 6.4 of the Master service Agreement.

# **2.6. Coordination**

**property name supplied.** 

All project management related communication from the Customer will be directed via pair as point of contact in MCA acting as Delivery Manager tandem identified below. The Delivery Managers will work with the Customer and leads to coordinate and integrate with other streams, produce an estimate of the time required for each change order (changes in scope, deliverables and skills) and determine whether additional resources are required. Any such changes shall be agreed in writing between the parties in accordance with section 4.

Project Manager contact details:

Sambhav Jain E-mail: Sambhav_Jain@MCA.com

Vikas Kalkhanday E-mail : vikas@MCA.com

# 3. **Customer Responsibilities**

The Customer shall:

- provide such information related to the Services as may be reasonably requested by MCA;
- provide MCA with access to, and the rights to use, the Customer's software, infrastructure and networks (including any necessary source code) and shall ensure that it obtains such third-party licenses as are necessary for MCA to perform the services;
- ensure that any relevant third-party suppliers cooperate reasonably with MCA;
- notify MCA of changes to the technical / IT environment made by the Customer or its other suppliers, where such technical changes may have an impact on the Services; and
- ensure the availability of relevant subject matter experts from the Customer and other third parties at the agreed times and locations, in particular to ensure the availability of the necessary Customer personnel to introduce MCA staff to Customer's s technology and relevant technological analysis together with the MCA team;
- ensure the teams have access to relevant LPRO systems, documentation and environments;
- provide any external dependency identified and agreed in writing that is outside of the control of MCA which is deemed by MCA to be the Customer's responsibility to manage;
- ensure that the Customer and its internal team will feed back, sign off or otherwise progress the project as per the agreed timelines during the project;
- Customer will ensure that the MCA technical consultant will get feedback from the customer's enterprise

**property name supplied.** 

architect within a business day;

- Customer will ensure that the relevant technical leads of teams are available to provide information regarding the existing technology stack and other backend systems. All queries related to Customer's existing technology stack and backend systems will be responded within a business day.
- Customer will ensure that the backlog for each individual team is available in advance for at least one mission board period(2 months) for the team capacity.
- Customer will provide a full-time product owner for the squad. All the user stories will have Cleary defined DoR(definition of ready) and DoD(definition of done) before being planned for the sprints in a mission board.
- Customer will ensure that the UAT testers are available from the customer to accept the user story(deliverable) and will complete UAT within the Sprint.

The Customer acknowledges and agrees that to the extent that the Customer fails or delays in complying with Customer's obligations under this SOW and the Agreement: (i) MCA shall not be liable for any resulting failure or delay in the performance of the Services; (ii) this may result in an increase in the costs of the project.

# 4. **Change Management**

If Customer wishes to propose changes to the Services, Customer shall notify MCA of the proposed changes in a change request. Following receipt of such request, MCA will provide an impact analysis to the Customer identifying the impact on the SOW of such changes in terms of (as applicable) costs, deliverables completion dates, functionality and performance. If both Parties agree in respect of the changes required, then the Parties will execute an appropriate change order.

# 5. **Commercial Terms**

# **5.1. Fees and Costs**

Payment for Services performed under this SOW will be on a Squad as a service model (Managed Capacity with delivery obligation) basis in accordance with the envisioned team composition and the daily rates specified in section 6. The team will be dedicated with fixed composition, capable of enhancing the search infrastructure as custom product development. The team will deliver an agreed upon story points or deliverables based on the capacity. The story points/ deliverables must be mutually agreed before every iteration and Sprint.

# **5.2. Invoicing**

MCA will invoice the Customer at the end of each calendar month based on the capacity of the team. For the first month of the iteration, the customer can withhold 5% of the invoice and for the second month invoice, if the team has met its obligations (within 10% deviation) then the customer is obliged to pay the full amount of invoice and the 5% from pervious month if the amount was withheld. If the team failed to deliver the agreed deliverables within the 10% deviation then customer will pay the withheld amount upon completion of deliverables. MCA will provide written reports, showing Story Point utilization/ deliverable completion status specific to the project or product, in a mutually agreed format on a monthly basis, and will maintain it throughout the term of the SOW.

For the avoidance of doubt, MCA is entitled to invoice the customer only for the agreed User stories / Deliverables, which have been accepted by the Customer.

### **5.3. Cost assumptions**

All travel expenses, need to be priorly approved in writing by Customer and will be charged to Customer on a passthrough basis.

14/12/2015 16:5407/04/2015 10:29:006/29/2009 4:22:00 PM**Error!** No **property name supplied.** 

# **5.4. Payment Terms**

MCA will invoice the Customer for the Services and any associated expenses incurred during the previous month in accordance with the terms of the Agreement. The invoicing address and details are as follows:

LPRO Parts Holding nv

Sa grada familia

Barcelona, Spain

### **5.5. Key Contacts**

The Contact persons for each of the Parties regarding this SOW and their contact information are:

For Customer: Nimesh sucks E-mail : nimesh.sucks@LPRO.com

Phone: +1 404 404112 1

MCA: Great Gambler E-mail: great_gambler@mca.com

## 6. **Resource Plan**

The resource plan is separated in two phases: Onboarding Workshop and Definition Phase:

**property name supplied.** 

14/12/2015 16:5407/04/2015 10:29:006/29/2009 4:22:00 PM**Error!** No

| Position | Location | Estimated | End date | Availability |
| --- | --- | --- | --- | --- |
|  |  | Start date |  |  |
| Data Analytics | UK |  | 31/12/2022 | 20% |
| Consulting Consultant |  | 13/06/2022 |  |  |
| Data and Analytics | Germany |  | 31/12/2022 | 25% |
| Technology Director |  | 13/06/2022 |  |  |
| Technology Solutions | Georgia |  |  | 20% |
| Director |  |  |  |  |
| Search Consultant | UK | 13/06/2022 | 31/12/2022 | 100% |
| Solution Architect | CEE |  | 31/12/2022 | 100% |
|  |  | 13/06/2022 |  |  |
| Data & Analytics Consultant | CEE | 13/06/2022 | 31/12/2022 | 50% |
| Lead Business Analyst |  |  |  | 100% |
|  | CEE | 13/06/2022 | 31/12/2022 |  |
| Lead Search Engineer | CEE | 13/06/2022 | 31/12/2022 | 100% |
| Lead ETL Developer | CEE | 13/06/2022 | 31/12/2022 | 100% |
| Lead DevOps Engineer | CEE |  | 31/12/2022 | 100% |
|  |  | 13/06/2022 |  |  |
| Frontend Engineer | CEE | 13/06/2022 | 31/12/2022 | 100% |
| Search Engineer | CIS | 25/07/2022 | 31/12/2022 | 100% |
| Search Engineer | CIS | 25/07/2022 | 31/12/2022 | 100% |
| ETL Developer | CEE | 25/07/2022 | 31/12/2022 | 100% |
| Lead QA Engineer | CEE | 25/07/2022 | 31/12/2022 | 100% |
| QA Engineer | CEE | 25/07/2022 | 31/12/2022 | 100% |
| Backend Engineer | CEE | 25/07/2022 | 31/12/2022 | 100% |
| DevOps Engineer | CEE | 25/07/2022 | 31/12/2022 | 100% |
| Project Manager | CEE |  | 31/12/2022 | 50% |
|  |  | 13/06/2022 |  |  |

## 7. **General**

This SOW and the Agreement constitute the entire agreement between the Parties and supersede and extinguish all previous agreements, promises, assurances, warranties, representations and understandings between them, whether written or oral, relating to their subject matter.

14/12/2015 16:5407/04/2015 10:29:006/29/2009 4:22:00 PM**Error!** No

**property name supplied.** 

Electronic copies of this SOW have the same strength of law and are equally legally binding as signed originals.

By signing below, each party represents it has read this Statement of Work, understands it, and agrees to be bound by it as of the SOW Effective Date.

MCA Systems Netherlands B.V. (Belgium Branch) LPRO Parts Holding NV

| By: | (Signature) | By: | (Signature) |
| --- | --- | --- | --- |
| Name: |  | Name: |  |
| Title: |  | Title: |  |
| Date: |  | Date: |  |

**Title**: **Statement of Work for provision of services Confidential**  14/12/2015 16:5407/04/2015 10:29:006/29/2009 4:22:00 PM**Error!** No

**property name supplied.** 

# **APPENDIX A: Definitions**

# 1. **Definitions of Done (applicable for software delivery only)**

- 1) All acceptance criteria met and fulfilled and checked by Product Owner
- 2) 100 % feature completion and tested by QA
- 3) All Unit Tests where applicable are run and green on development instance
- 4) Error cases/exceptions and edge cases are handled, and appropriate notification are given to the user
- 5) The user story code compiles successfully as part of daily build
- 6) No build errors
- 7) Shippable Software artifacts provided / uploaded to artifact repository
- 8) All code changes have been verified by Code Reviews
- 9) Zero-Defect-Story: No new known Blocker and Critical issues introduced with the story and no known regression introduced
- 10) User story is compliant with the structural code quality KPIs as defined by the Customer

# 2. **Definition of Ready ("DOR") (applicable for software delivery only)**

A user story is considered ready for implementation if it complies with the following "Definition of Ready" criteria:

- 1) Story must have a clear answer on question "why?" from Business standpoint.
- 2) User story is of adequate granularity
- 3) Acceptance criteria must exist and be understood by the team, executable and not ambiguous
- 4) User story should contain the final wireframe and visual design (if needed) and it shouldn't change along the sprint.
- 5) Acceptance criteria must be reviewed from Product Owners side.
- 6) Story has been estimated by the team (high level estimates) and thus, team understand how to implement the story.
- 7) User Story should pass Development (BE, FE) and QA review.

