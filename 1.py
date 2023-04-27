heatmap = go.Heatmap(
    x=df['x'],
    y=df['y'],
    z=df['value']
)

# Создаем объект фигуры
fig = go.Figure(data=[heatmap])

# Настраиваем макет
fig.update_layout(
    title='Пример интерактивной тепловой карты для датафрейма',
    xaxis=dict(title='X Axis Title'),
    yaxis=dict(title='Y Axis Title'),
)

# Отображаем фигуру
fig.show()