#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


df= pd.read_csv('hotel_bookings.csv')
df.head()


# In[13]:


# Understanding the data
df.shape


# In[14]:


pd.set_option('display.max_columns',32)
df.head()


# In[15]:


df.columns


# In[16]:


df.nunique()


# In[17]:


df['hotel'].value_counts()


# In[18]:


df['meal'].value_counts()


# In[19]:


df['market_segment'].value_counts()


# In[20]:


df['distribution_channel'].value_counts()


# In[21]:


df['deposit_type'].value_counts()


# In[22]:


df['customer_type'].value_counts()


# In[23]:


df['total_of_special_requests'].value_counts()


# In[24]:


sns.countplot(data=df,x='hotel')


# In[25]:


sns.countplot(data=df,x='is_canceled',hue='is_repeated_guest')


# In[26]:


sns.countplot(data=df,x='hotel',hue='is_canceled')


# In[27]:


# Checking for null values
df.isnull().values.any()


# In[28]:


df.isnull().sum()


# In[29]:


#Replacing missing Values with 0
df.fillna(0,inplace=True)


# In[30]:


df.isnull().sum()


# In[31]:


#Meal contains values "Undefined", which is equal to SC
df["meal"].replace("Undefined","SC", inplace=True)


# In[32]:


df["meal"].unique()


# In[33]:


Subset=df[(df['children']==0) & (df['adults']==0) & (df['babies']==0)]


# In[34]:


Subset[['adults','babies','children']]


# In[35]:


type(Subset)


# In[36]:


Delete=(df['children']==0) & (df['adults']==0) & (df['babies']==0)


# In[37]:


type(Delete)


# In[38]:


Delete


# In[39]:


data=df[~Delete]


# In[40]:


data.head()


# In[41]:


Subset=data[(data['children']==0) & (data['adults']==0) & (data['babies']==0)]


# In[42]:


Subset


# In[43]:


data.shape


# In[44]:


119390-119210


# In[45]:


data.to_csv('Updated_Hotel_Booking.csv', index=False)


# In[46]:


guest_country=data[data['is_canceled']==0  ]['country'].value_counts().reset_index()
guest_country.columns=['country','Number of guests']


# In[47]:


guest_country


# In[48]:


import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.express as px


# In[49]:


total_guests = guest_country["Number of guests"].sum()
print(total_guests)


# In[50]:


guest_country["Guests in %"] = round(guest_country["Number of guests"] / total_guests * 100, 2)
guest_country


# In[51]:


trace= go.Bar(
    x=guest_country["country"],
    y=guest_country['Number of guests'],
    marker=dict(color='#CD7F32') 
)
data1 = [ trace]
layout = go.Layout(
    title='Guests by Country'
)
fig = go.Figure(data=data1, layout=layout)
pyo.plot(fig)


# In[52]:


map_guest = px.choropleth(guest_country,
                    locations=guest_country['country'],
                    color=guest_country['Number of guests'], 
                    hover_name=guest_country['country'], 
                    title="Home country of guests")
map_guest.show()


# In[53]:


resort = data[(data["hotel"] == "Resort Hotel") & (data["is_canceled"] == 0)]
city = data[(data["hotel"] == "City Hotel") & (data["is_canceled"] == 0)]


# In[54]:


resort


# In[55]:


resort_hotel=resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
resort_hotel


# In[56]:


city_hotel=city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel


# In[57]:


final=resort_hotel.merge(city_hotel,on='arrival_date_month')
final.columns=['month','price_for_resort','price_for_city_hotel']
final


# In[58]:


import sort_dataframeby_monthorweek as sd
final=sd.Sort_Dataframeby_Month(df=final,monthcolumnname='month')
final


# In[65]:


px.line(final, x='month',
        y=['price_for_resort','price_for_city_hotel'],
        title='Room price per night over the Months')


# In[66]:


#### How Much Do Guests Pay For A Room Per Night?  
df['reserved_room_type'].unique()


# In[67]:


data["adr_Updated"]=data["adr"]/(data["adults"]+data["children"])
data


# In[68]:


data["adr_Updated"]=data["adr"]/(data["adults"]+data["children"])
valid_guest= data.loc[data["is_canceled"] == 0]
prices = valid_guest[["hotel", "reserved_room_type", "adr_Updated"]].sort_values("reserved_room_type")

plt.figure(figsize=(12, 8))
sns.boxplot(x="reserved_room_type",
            y="adr_Updated",
            hue="hotel",
            data=prices
           )
plt.title("Price of room types per night and person", fontsize=16)
plt.xlabel("Room type", fontsize=16)
plt.ylabel("Price [EUR]", fontsize=16)

plt.ylim(0, 160)
plt.show()


# In[69]:


prices_C=prices[prices['reserved_room_type']=='C']
prices_C


# In[70]:


prices_City=prices_C[prices_C['hotel']=='City Hotel']
prices_Resort=prices_C[prices_C['hotel']=='Resort Hotel']
prices_Resort


# In[71]:


prices_City


# In[72]:


prices_Resort.describe()


# In[73]:


#### How long do people stay at the hotels? 
df3=data[data['is_canceled']==0]
df3["total_nights"] = df3["stays_in_weekend_nights"] + df3["stays_in_week_nights"]


# In[74]:


df3


# In[75]:


df4=df3[['total_nights','hotel','is_canceled']]
df4


# In[76]:


hotel_stay=df4.groupby(['total_nights','hotel']).agg('count').reset_index()

hotel_stay


# In[77]:


hotel_stay=hotel_stay.rename(columns={'is_canceled':'Number of stays'})
hotel_stay.head()


# In[78]:


hotel_stay_r=hotel_stay[hotel_stay['hotel']=='Resort Hotel']
hotel_stay_r


# In[79]:


hotel_stay_c=hotel_stay[hotel_stay['hotel']=='City Hotel']
hotel_stay_c


# In[80]:


trace = go.Bar(
    x=hotel_stay_r["total_nights"],
    y=hotel_stay_r["Number of stays"],
    name='Resort Stay'
    )

trace1=go.Bar(
    x=hotel_stay_c["total_nights"],
    y=hotel_stay_c["Number of stays"],
    name='City stay'
    )


data5 = [trace,trace1]
layout = go.Layout(
    title='Total Number of stays by Guest'
)
fig = go.Figure(data=data5, layout=layout)
pyo.plot(fig)


# In[81]:


#### Bookings by market segment 
segments=data["market_segment"].value_counts()
segments


# In[82]:


segments=data["market_segment"].value_counts()

# pie plot
fig = px.pie(segments,
             values=segments.values,
             names=segments.index,
             title="Bookings per market segment",
             template="seaborn")
fig.update_traces(rotation=-90, textinfo="percent+label")
fig.show()


# In[83]:


plt.figure(figsize=(12, 8))
sns.barplot(x="market_segment",
            y="adr_Updated",
            hue="reserved_room_type",
            data=data,
            ci=None)
plt.title("ADR by market segment and room type", fontsize=16)
plt.xlabel("Market segment", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("ADR per person [EUR]", fontsize=16)
plt.legend(loc="upper left")
plt.show()


# In[84]:


#### How many bookings were canceled?
Cancel=data['is_canceled']==1


# In[85]:


cancel=Cancel.sum()


# In[86]:


resort_cancelation = data.loc[data["hotel"] == "Resort Hotel"]["is_canceled"].sum()
city_cancelation = data.loc[data["hotel"] == "City Hotel"]["is_canceled"].sum()


# In[87]:


resort_cancelation


# In[88]:


city_cancelation


# In[89]:


print(f"Total Booking Cancelled : {cancel} . ")
print(f"Total Resort Hotel Booking Cancelled : {resort_cancelation} . ")
print(f"Total City Hotel Booking Cancelled : {city_cancelation} . ")


# In[90]:


#### Which month have the highest number of cancelations?
res_book_per_month = data.loc[(data["hotel"] == "Resort Hotel")].groupby("arrival_date_month")["hotel"].count()
res_cancel_per_month = data.loc[(data["hotel"] == "Resort Hotel")].groupby("arrival_date_month")["is_canceled"].sum()

cty_book_per_month = data.loc[(data["hotel"] == "City Hotel")].groupby("arrival_date_month")["hotel"].count()
cty_cancel_per_month = data.loc[(data["hotel"] == "City Hotel")].groupby("arrival_date_month")["is_canceled"].sum()

res_cancel_data = pd.DataFrame({"Hotel": "Resort Hotel",
                                "Month": list(res_book_per_month.index),
                                "Bookings": list(res_book_per_month.values),
                                "Cancelations": list(res_cancel_per_month.values)})
cty_cancel_data = pd.DataFrame({"Hotel": "City Hotel",
                                "Month": list(cty_book_per_month.index),
                                "Bookings": list(cty_book_per_month.values),
                                "Cancelations": list(cty_cancel_per_month.values)})


# In[91]:


res_cancel_data


# In[92]:


import sort_dataframeby_monthorweek as sd
res_cancel_data=sd.Sort_Dataframeby_Month(df=res_cancel_data,monthcolumnname='Month')
res_cancel_data


# In[93]:


import sort_dataframeby_monthorweek as sd
cty_cancel_data=sd.Sort_Dataframeby_Month(df=cty_cancel_data,monthcolumnname='Month')
cty_cancel_data


# In[94]:


plt.figure(figsize=(12, 8))

trace = go.Bar(
    x=res_cancel_data["Month"],
    y=res_cancel_data["Cancelations"],
    name="Rst Cancelled"
    )
trace1 = go.Bar(
    x=cty_cancel_data["Month"],
    y=cty_cancel_data["Cancelations"],
    name="Cty Cancelled"
    )


data6 = [trace,trace1]
layout = go.Layout(
    title='Total Number of stays by Guest'
)
fig = go.Figure(data=data6, layout=layout)
pyo.plot(fig)


# In[ ]:




