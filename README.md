# sagemaker_end_to_end_ml

Repository for demo of AWS SageMaker for end-to-end ML, including setup, training, deployment, and bias detection.

From How To video: https://youtu.be/HtnwFBzV1r4



dvc init

dvc repro

dvc dag







      - name: Pull latest image
        run: |
          docker pull ${{ secrets.DOCKER_USERNAME }}/bank_demo:latest

      - name: Stop and remove old container if exists
        run: |
          docker ps -q --filter "name=texts" | grep -q . && docker stop texts && docker rm -fv texts || true

      - name: Run Docker container
        run: |
          docker run -d -p 8501:8501 ${{ secrets.DOCKER_USERNAME }}/bank_demo:latest

      - name: Clean previous images
        run: docker system prune -f
