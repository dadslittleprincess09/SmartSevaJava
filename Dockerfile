FROM eclipse-temurin:21-jdk AS build
WORKDIR /app

COPY . .

# Give executable permission to mvnw (THIS FIXES THE ERROR)
RUN chmod +x mvnw

RUN ./mvnw clean package -DskipTests

FROM eclipse-temurin:21-jdk
WORKDIR /app

COPY --from=build /app/target/*.jar app.jar

EXPOSE 8080

ENTRYPOINT ["sh", "-c", "java -jar app.jar --server.port=$PORT"]
